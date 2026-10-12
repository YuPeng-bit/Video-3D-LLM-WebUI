"""Official evaluation metrics, re-used verbatim from the upstream repository.

``llava/eval/eval_<dataset>.py`` are corpus-level scripts.  The WebUI needs a
score for *every single* generated answer, so this module reuses the very same
scorer classes and reproduces the official preprocessing of predictions and
ground truth.  Per-sample values are identical to the values those samples get
inside the official corpus run:

* CIDEr uses the full official reference corpus for document frequency / length
  normalisation (per-item score only depends on the references).
* BLEU (``option='closest'``), METEOR and ROUGE-L are per-item metrics.
* ScanRefer / Multi3DRefer use the official 3D IoU helpers of
  ``llava/eval/box_utils.py`` (Multi3DRefer also keeps the official 0.4
  cumulative-softmax threshold and Hungarian matching).
"""

from __future__ import annotations

import os
import re
import sys
import threading
from typing import Dict, List, Optional, Sequence

import contextlib
import io

import numpy as np

from . import config

if config.REPO_ROOT not in sys.path:
    sys.path.insert(0, config.REPO_ROOT)


# --------------------------------------------------------------------------
# official helpers
# --------------------------------------------------------------------------
def _cider_scorer_cls():
    from llava.eval.caption_eval.cider.cider_scorer import CiderScorer

    return CiderScorer


def _caption_metrics():
    from llava.eval.caption_eval.bleu.bleu import Bleu
    from llava.eval.caption_eval.rouge.rouge import Rouge

    return Bleu, Rouge


def _box_utils():
    from llava.eval.box_utils import box3d_iou, get_3d_box_corners

    return box3d_iou, get_3d_box_corners


# --------------------------------------------------------------------------
# text normalisation used by the official scripts
# --------------------------------------------------------------------------
def scanqa_normalize_id(sample_id: str) -> str:
    """``scanqa_val-scene0011-0_3`` -> ``scanqa_val-scene0011-0_0`` (official)."""
    parts = str(sample_id).split("_")
    return "_".join(parts[:-1] + ["0"])


def scan2cap_style(text: str) -> str:
    return "sos " + str(text).replace(".", " . ").replace(",", " , ").lower() + " eos"


_CLEAN_ANSWER_RULES = [
    (r"[ ]+$", ""),
    (r"^[ ]+", ""),
    (r" {2,}", " "),
    (r"\.[ ]{2,}", ". "),
    (r"[^a-zA-Z0-9,\'\s\-:]+", ""),
    ("\u00e7", "c"),
    ("\u2019", "'"),
    (r"\bletf\b", "left"),
    (r"\blet\b", "left"),
    (r"\btehre\b", "there"),
    (r"\brigth\b", "right"),
    (r"\brght\b", "right"),
    (r"\bbehine\b", "behind"),
    (r"\btv\b", "TV"),
    (r"\bchai\b", "chair"),
    (r"\bwasing\b", "washing"),
    (r"\bwaslked\b", "walked"),
    (r"\boclock\b", "o\'clock"),
    (r"\bo\'[ ]+clock\b", "o\'clock"),
    (r"\b0\b", "zero"),
    (r"\bnone\b", "zero"),
    (r"\b1\b", "one"),
    (r"\b2\b", "two"),
    (r"\b3\b", "three"),
    (r"\b4\b", "four"),
    (r"\b5\b", "five"),
    (r"\b6\b", "six"),
    (r"\b7\b", "seven"),
    (r"\b8\b", "eight"),
    (r"\b9\b", "nine"),
    (r"\b10\b", "ten"),
    (r"\b11\b", "eleven"),
    (r"\b12\b", "twelve"),
    (r"\b13\b", "thirteen"),
    (r"\b14\b", "fourteen"),
    (r"\b15\b", "fifteen"),
    (r"\b16\b", "sixteen"),
    (r"\b17\b", "seventeen"),
    (r"\b18\b", "eighteen"),
    (r"\b19\b", "nineteen"),
    (r"\b20\b", "twenty"),
    (r"\b23\b", "twenty-three"),
    (r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>"),
    (r"\ba\b ([a-zA-Z]+)", r"\g<1>"),
    (r"\ban\b ([a-zA-Z]+)", r"\g<1>"),
    (r"\bthe\b ([a-zA-Z]+)", r"\g<1>"),
    (r"\bbackwards\b", "backward"),
]


def clean_answer(data: str) -> str:
    """``llava/eval/eval_sqa3d.py::clean_answer`` (verbatim)."""
    data = str(data).lower()
    for pattern, repl in _CLEAN_ANSWER_RULES:
        data = re.sub(pattern, repl, data)
    return data


# --------------------------------------------------------------------------
# metric engine
# --------------------------------------------------------------------------
class MetricEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._q_cache: Dict[str, List[dict]] = {}
        self._corpus_cache: Dict[str, List[tuple]] = {}
        self._cider_ctx: Dict[str, tuple] = {}
        self._meteor = None

    # -- corpus ---------------------------------------------------------
    def _questions(self, key: str) -> List[dict]:
        with self._lock:
            if key not in self._q_cache:
                with open(config.abs_question_file(key), "r", encoding="utf-8") as fh:
                    self._q_cache[key] = __import__("json").load(fh)
            return self._q_cache[key]

    def corpus(self, key: str) -> List[tuple]:
        """Ordered ``[(sample_id, references), ...]`` exactly as the official
        evaluation script builds them."""
        with self._lock:
            if key in self._corpus_cache:
                return self._corpus_cache[key]
            records = self._questions(key)
            if key == "scanqa":
                ordered: Dict[str, List[str]] = {}
                for rec in records:
                    sid = scanqa_normalize_id(rec.get("id"))
                    ordered[sid] = list((rec.get("metadata") or {}).get("answers") or [])
                items = list(ordered.items())
            elif key == "scan2cap":
                items = []
                for rec in records:
                    refs = rec.get("annotations")
                    if not refs:
                        conv = rec.get("conversations") or []
                        refs = [conv[-1].get("value")] if conv else []
                    items.append((rec.get("id"), [scan2cap_style(r) for r in refs]))
            else:
                items = []
            self._corpus_cache[key] = items
            return items

    def corpus_preview(self, key: str) -> List[tuple]:
        return self.corpus(key)

    # -- per sample -----------------------------------------------------
    def evaluate(self, key: str, record: dict, prediction) -> dict:
        metric_set = config.dataset_info(key)["metric_set"]
        if metric_set in ("scanqa", "scan2cap"):
            return self._text_metrics(key, record, prediction)
        if metric_set == "sqa3d":
            return self._sqa3d(record, prediction)
        if metric_set == "scanrefer":
            return self._scanrefer(record, prediction)
        if metric_set == "multi3drefer":
            return self._multi3drefer(record, prediction)
        raise KeyError(metric_set)

    # -- ScanQA / Scan2Cap ---------------------------------------------
    def _meteor_instance(self):
        with self._lock:
            if self._meteor is None:
                from llava.eval.caption_eval.meteor.meteor import Meteor

                self._meteor = Meteor()
            return self._meteor

    def _text_metrics(self, key: str, record: dict, prediction: str) -> dict:
        if key == "scanqa":
            sid = scanqa_normalize_id(record.get("id"))
            refs = list((record.get("metadata") or {}).get("answers") or [])
            hypo = str(prediction).rstrip(".")
            em = int(str(prediction) in refs)
        else:
            sid = record.get("id")
            refs = record.get("annotations") or []
            if not refs:
                conv = record.get("conversations") or []
                refs = [conv[-1].get("value")] if conv else []
            refs = [scan2cap_style(r) for r in refs]
            hypo = scan2cap_style(prediction)
            em = None

        if not refs:
            return {"error": "no ground-truth references for this sample"}

        out: Dict[str, object] = {}
        out["CIDEr"] = round(self._cider_single(key, sid, hypo, refs) * 100, 4)
        bleu_k, rouge_l = self._bleu_rouge_single(sid, hypo, refs)
        for k, v in enumerate(bleu_k, start=1):
            out[f"BLEU-{k}"] = round(v * 100, 4)
        out["METEOR"] = round(self._meteor_single(sid, hypo, refs) * 100, 4)
        out["ROUGE-L"] = round(rouge_l * 100, 4)
        if em is not None:
            out["EM"] = em
        out["_gt_normalized"] = refs
        out["_pred_normalized"] = hypo
        return out

    def _cider_context(self, key: str):
        """Cached (document_frequency, log corpus size) of the official corpus.

        CIDEr's per-item score only depends on the reference corpus statistics
        and on that item's own hypothesis/references, so caching these two
        values makes the per-sample score exact *and* O(1).
        """
        with self._lock:
            if key not in self._cider_ctx:
                CiderScorer = _cider_scorer_cls()
                scorer = CiderScorer(n=4, sigma=6.0)
                for _sid, refs in self.corpus(key):
                    scorer += (refs[0], refs)
                scorer.compute_doc_freq()
                self._cider_ctx[key] = (
                    scorer.document_frequency,
                    float(np.log(float(len(scorer.crefs)))),
                )
            return self._cider_ctx[key]

    def _cider_single(self, key: str, sid, hypo: str, refs: List[str]) -> float:
        document_frequency, ref_len = self._cider_context(key)
        corpus_size = len(self.corpus(key))
        CiderScorer = _cider_scorer_cls()
        scorer = CiderScorer(n=4, sigma=6.0)
        scorer += (hypo, refs)
        # pad the reference list so that ``compute_cider`` sees the official
        # corpus size (it derives ``ref_len`` from ``len(self.crefs)``) while
        # ``zip`` still only scores our single hypothesis
        if len(scorer.crefs) < corpus_size:
            scorer.crefs = list(scorer.crefs) + [None] * (corpus_size - len(scorer.crefs))
        scorer.document_frequency = document_frequency
        scorer.ref_len = ref_len
        scores = scorer.compute_cider()
        return float(scores[0])

    def _bleu_rouge_single(self, sid, hypo: str, refs: List[str]):
        Bleu, Rouge = _caption_metrics()
        bleu = Bleu()
        with _quiet():
            _score, per = bleu.compute_score({sid: list(refs)}, {sid: [hypo]})
        bleu_k = [float(per[k][0]) for k in range(4)]
        rouge = Rouge()
        with _quiet():
            _r, rper = rouge.compute_score({sid: list(refs)}, {sid: [hypo]})
        return bleu_k, float(rper[0])

    def _meteor_single(self, sid, hypo: str, refs: List[str]) -> float:
        meteor = self._meteor_instance()
        with self._lock, _quiet():
            _score, per = meteor.compute_score({sid: list(refs)}, {sid: [hypo]})
        return float(per[0])

    # -- SQA3D ----------------------------------------------------------
    def _sqa3d(self, record: dict, prediction: str) -> dict:
        conv = record.get("conversations") or []
        gt = conv[-1].get("value") if conv else ""
        pred_c = clean_answer(prediction)
        gt_c = clean_answer(gt)
        return {
            "EM(acc)": int(pred_c == gt_c),
            "_pred_normalized": pred_c,
            "_gt_normalized": gt_c,
            "_gt_raw": gt,
        }

    # -- ScanRefer ------------------------------------------------------
    def _scanrefer(self, record: dict, prediction) -> dict:
        box3d_iou, get_3d_box_corners = _box_utils()
        pred = _as_box(prediction)
        gt = _as_box(record.get("box"))
        if pred is None or gt is None:
            return {"error": "could not parse a 3D box (need 6 numbers: cx cy cz sx sy sz)"}
        iou = float(box3d_iou(get_3d_box_corners(gt[:3], gt[3:]), get_3d_box_corners(pred[:3], pred[3:])))
        return {
            "IoU": round(iou, 4),
            "IoU@0.25": int(iou >= 0.25),
            "IoU@0.5": int(iou >= 0.5),
            "_pred_box": pred,
            "_gt_box": gt,
        }

    # -- Multi3DRefer ---------------------------------------------------
    def _multi3drefer(self, record: dict, prediction) -> dict:
        threshold = 0.4  # official default of eval_multi3drefer.py
        scores, objects, pred_boxes = _select_objects(prediction, threshold)
        gt_boxes = record.get("box") or []
        f1_25, f1_50 = evaluate_one_query(pred_boxes, gt_boxes)
        gt = record.get("box") or []
        return {
            "F1@0.25": round(float(f1_25), 4),
            "F1@0.5": round(float(f1_50), 4),
            "_threshold": threshold,
            "_n_pred": len(pred_boxes),
            "_n_gt": len(gt),
            "_pred_boxes": pred_boxes,
            "_gt_boxes": gt,
            "_scores": None if scores is None else [float(s) for s in scores],
        }

    # -- corpus level (verification) ------------------------------------
    def corpus_metrics(self, key: str, predictions: Dict[object, object], gt_override: Optional[dict] = None) -> dict:
        """Official corpus-level metrics for a {sample_id: prediction} mapping."""
        metric_set = config.dataset_info(key)["metric_set"]
        if metric_set == "scanqa":
            gts, res = {}, {}
            id2labels = {}
            for rec in self._questions(key):
                id2labels[rec.get("id")] = (rec.get("metadata") or {}).get("answers")
            n_correct = 0
            total = 0
            for sid_raw, pred in predictions.items():
                sid = scanqa_normalize_id(sid_raw)
                gts[sid] = id2labels[sid_raw]
                res[sid] = [str(pred).rstrip(".")]
                if str(pred) in id2labels[sid_raw]:
                    n_correct += 1
                total += 1
            return {**_caption_corpus(gts, res), "EM": n_correct / max(1, total), "count": len(gts)}
        if metric_set == "scan2cap":
            gts, res = {}, {}
            id2ann = {}
            if gt_override is not None:
                id2ann = {sid: [scan2cap_style(r) for r in refs] for sid, refs in gt_override.items()}
            else:
                for rec in self._questions(key):
                    refs = rec.get("annotations") or [(rec.get("conversations") or [{}])[-1].get("value")]
                    id2ann[rec.get("id")] = [scan2cap_style(r) for r in refs]
            for sid, pred in predictions.items():
                gts[sid] = id2ann[sid]
                res[sid] = [scan2cap_style(pred)]
            return {**_caption_corpus(gts, res), "count": len(gts)}
        if metric_set == "sqa3d":
            conv = {rec.get("id"): (rec.get("conversations") or [{}])[-1].get("value") for rec in self._questions(key)}
            correct = [int(clean_answer(p) == clean_answer(conv[sid])) for sid, p in predictions.items()]
            return {"EM(acc)": 100 * float(np.mean(correct)), "count": len(correct)}
        if metric_set == "scanrefer":
            box3d_iou, get_3d_box_corners = _box_utils()
            gts = {rec.get("id"): rec.get("box") for rec in self._questions(key)}
            a25, a50 = [], []
            for sid, pred in predictions.items():
                p, g = _as_box(pred), _as_box(gts[sid])
                iou = float(box3d_iou(get_3d_box_corners(g[:3], g[3:]), get_3d_box_corners(p[:3], p[3:])))
                a25.append(iou >= 0.25)
                a50.append(iou >= 0.5)
            return {
                "IoU@0.25": 100 * float(np.mean(a25)),
                "IoU@0.5": 100 * float(np.mean(a50)),
                "count": len(a25),
            }
        if metric_set == "multi3drefer":
            gts = {rec.get("id"): rec.get("box") for rec in self._questions(key)}
            f25, f50 = [], []
            for sid, pred in predictions.items():
                _, _, boxes = _select_objects(pred, 0.4)
                a, b = evaluate_one_query(boxes, gts[sid] or [])
                f25.append(a)
                f50.append(b)
            return {
                "F1@0.25": float(np.mean(f25)),
                "F1@0.5": float(np.mean(f50)),
                "count": len(f25),
            }
        raise KeyError(metric_set)


_QUIET_LOCK = threading.Lock()


@contextlib.contextmanager
def _quiet():
    """The upstream BLEU scorer prints debug lines; keep the server log clean."""
    with _QUIET_LOCK:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            yield buf


def _caption_corpus(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> dict:
    from llava.eval.caption_eval.bleu.bleu import Bleu
    from llava.eval.caption_eval.cider.cider import Cider
    from llava.eval.caption_eval.meteor.meteor import Meteor
    from llava.eval.caption_eval.rouge.rouge import Rouge

    out: Dict[str, float] = {}
    cider_score = Cider().compute_score(gts, res)
    out["CIDEr"] = float(cider_score[0]) * 100
    bleu = Bleu()
    with _quiet():
        bleu_score = bleu.compute_score(gts, res)
    for k, v in enumerate(bleu_score[0], start=1):
        out[f"BLEU-{k}"] = float(v) * 100
    meteor = Meteor()
    with _quiet():
        meteor_score = meteor.compute_score(gts, res)
    out["METEOR"] = float(meteor_score[0]) * 100
    rouge_score = Rouge().compute_score(gts, res)
    out["ROUGE-L"] = float(rouge_score[0]) * 100
    return out


# --------------------------------------------------------------------------
# grounding helpers (identical maths to llava/eval/eval_multi3drefer.py)
# --------------------------------------------------------------------------
def _as_box(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 6 and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flat: List[float] = []
        for v in value:
            if isinstance(v, (list, tuple)) and len(v) == 6:
                return [float(x) for x in v]
            if isinstance(v, (int, float)):
                flat.append(float(v))
        if len(flat) == 6:
            return flat
    if isinstance(value, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if len(nums) >= 6:
            return [float(x) for x in nums[:6]]
    return None


def _select_objects(prediction, threshold: float):
    """``eval_multi3drefer.py`` object selection with the official threshold."""
    if prediction is None:
        return None, None, []
    scores = prediction.get("scores") if isinstance(prediction, dict) else None
    objects = prediction.get("objects") if isinstance(prediction, dict) else None
    if scores is None or objects is None:
        boxes = prediction.get("boxes") if isinstance(prediction, dict) else prediction
        return None, None, [list(map(float, b)) for b in (boxes or [])]

    import torch

    s = torch.tensor(scores, dtype=torch.float32)
    objects = [list(map(float, o)) for o in objects]
    selected: List[List[float]] = []
    if torch.max(s, dim=0)[1].item() != len(s) - 1:
        s = torch.nn.functional.softmax(s / 0.07)[:-1]
        sorted_scores, indices = torch.sort(s, descending=True)
        cur_sum = 0.0
        for idx, sc in zip(indices, sorted_scores):
            cur_sum += float(sc)
            selected.append(objects[int(idx)])
            if cur_sum >= threshold:
                break
    return scores, objects, selected


def evaluate_one_query(pred_info, gt_info):
    """Verbatim from ``llava/eval/eval_multi3drefer.py``."""
    from scipy.optimize import linear_sum_assignment

    box3d_iou, get_3d_box_corners = _box_utils()
    pred_bboxes_count = len(pred_info)
    gt_bboxes_count = len(gt_info)

    if pred_bboxes_count == 0 and gt_bboxes_count == 0:
        return 1, 1
    if pred_bboxes_count == 0 or gt_bboxes_count == 0:
        return 0, 0

    iou25_tp = 0
    iou50_tp = 0
    square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
    iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
    for i, pred_box in enumerate(pred_info):
        for j, gt_box in enumerate(gt_info):
            obj_center, obj_box_size = pred_box[:3], pred_box[3:]
            gt_center, gt_box_size = gt_box[:3], gt_box[3:]
            iou_matrix[i, j] = box3d_iou(
                get_3d_box_corners(obj_center, obj_box_size),
                get_3d_box_corners(gt_center, gt_box_size),
            )
    row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)
    for i in range(pred_bboxes_count):
        iou = iou_matrix[row_idx[i], col_idx[i]]
        if iou >= 0.25:
            iou25_tp += 1
        if iou >= 0.50:
            iou50_tp += 1
    iou25_f1_score = 2 * iou25_tp / (pred_bboxes_count + gt_bboxes_count)
    iou50_f1_score = 2 * iou50_tp / (pred_bboxes_count + gt_bboxes_count)
    return iou25_f1_score, iou50_f1_score


METRICS = MetricEngine()
