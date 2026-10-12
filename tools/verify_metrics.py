"""Validate ``app.metrics`` element-by-element against the official scorers.

No GPU required: this replays an existing official result file through

  1. the upstream script ``llava/eval/eval_<dataset>.py`` (the printed number),
  2. the official scorer classes over the full corpus (per-item arrays), and
  3. the WebUI metric engine (corpus level + per sample).

Step 3 must reproduce step 2 exactly and step 1 exactly.

Usage:
    python tools/verify_metrics.py                  # every dataset that has results
    python tools/verify_metrics.py scanqa scan2cap
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, metrics as M  # noqa: E402

RESULT_FILES = {
    "scanqa": "results/gt_structure_full/scanqa/baseline.jsonl",
    "scan2cap": "results/gt_structure_full/scan2cap/baseline.jsonl",
    "scanrefer": "results/gt_structure_full/scanrefer/baseline.jsonl",
    "multi3drefer": "results/gt_structure_full/multi3drefer/baseline.jsonl",
    "sqa3d": "results/gt_structure_full/sqa3d/baseline.jsonl",
}


def load_rows(key: str):
    path = os.path.join(config.REPO_ROOT, RESULT_FILES[key])
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, path


def prediction_of(key: str, row: dict):
    if key == "multi3drefer":
        return {"scores": row["scores"], "objects": row["objects"]}
    if key == "scanrefer":
        return row["pred_response"]
    return row["pred_response"]


def record_for(key: str, row: dict):
    """The record the WebUI would evaluate: for scanqa/sqa3d/scanrefer/
    multi3drefer the GT lives in the question file and the ids line up; the
    scan2cap baseline file was produced on a re-ordered question list, so we
    rebuild the record from the row itself."""
    if key == "scan2cap":
        refs = row["gt_response"]
        return {
            "id": row["sample_id"],
            "annotations": refs,
            "conversations": [
                {"from": "human", "value": (row.get("prompt") or "").split("\n")[-1]},
                {"from": "gpt", "value": refs[0] if refs else ""},
            ],
            "metadata": {"dataset": "scan2cap", "question_type": row.get("question_type")},
        }
    return _question_cache(key).get(row["sample_id"])


_QCACHE: dict = {}


def _question_cache(key: str):
    if key not in _QCACHE:
        with open(config.abs_question_file(key), "r", encoding="utf-8") as fh:
            _QCACHE[key] = {rec.get("id"): rec for rec in json.load(fh)}
    return _QCACHE[key]


def run_official(key: str, path: str):
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONPATH"] = config.REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, os.path.join(config.REPO_ROOT, config.dataset_info(key)["official_eval_script"]),
         "--input-file", path],
        capture_output=True, text=True, cwd=config.REPO_ROOT, env=env, timeout=7200,
    )
    return out.stdout.strip(), out.stderr.strip(), out.returncode


# --------------------------------------------------------------------------
def official_per_item(key: str, rows):
    """Per-item arrays straight from the upstream scorer classes."""
    from llava.eval.caption_eval.bleu.bleu import Bleu
    from llava.eval.caption_eval.cider.cider import Cider
    from llava.eval.caption_eval.meteor.meteor import Meteor
    from llava.eval.caption_eval.rouge.rouge import Rouge

    style = M.scan2cap_style
    if key == "scanqa":
        id2lab = {r.get("id"): (r.get("metadata") or {}).get("answers") for r in _question_cache(key).values()}
        gts, res, keys = {}, {}, []
        for row in rows:
            sid = M.scanqa_normalize_id(row["sample_id"])
            if sid in gts:
                continue
            keys.append(sid)
            gts[sid] = id2lab[row["sample_id"]]
            res[sid] = [str(row["pred_response"]).rstrip(".")]
    else:
        keys, gts, res = [], {}, {}
        for row in rows:
            sid = row["sample_id"]
            keys.append(sid)
            gts[sid] = [style(r) for r in row["gt_response"]]
            res[sid] = [style(row["pred_response"])]
    out = {}
    out["CIDEr"] = list(Cider().compute_score(gts, res)[1])
    bleu_per = Bleu().compute_score(gts, res)[1]
    for k in range(4):
        out[f"BLEU-{k+1}"] = list(bleu_per[k])
    out["METEOR"] = list(Meteor().compute_score(gts, res)[1])
    out["ROUGE-L"] = list(Rouge().compute_score(gts, res)[1])
    return keys, out


def main() -> int:
    keys = sys.argv[1:] or list(RESULT_FILES)
    worst = {}
    for key in keys:
        print("=" * 80)
        print(f"### {key}")
        rows, path = load_rows(key)
        if rows is None:
            print(f"  [skip] no result file: {RESULT_FILES[key]}")
            continue
        print(f"  replaying {len(rows)} predictions from {RESULT_FILES[key]}")

        stdout, stderr, rc = run_official(key, path)
        print(f"  --- upstream script (rc={rc}) ---")
        for line in stdout.splitlines():
            print("    " + line)

        preds = {row["sample_id"]: prediction_of(key, row) for row in rows}
        gt_override = None
        if key == "scan2cap":
            gt_override = {row["sample_id"]: row["gt_response"] for row in rows}
        corpus = M.METRICS.corpus_metrics(key, preds, gt_override=gt_override)
        print("  --- WebUI corpus level ---")
        print("    " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in corpus.items()}))

        # per-item comparison against the official scorer arrays
        if key in ("scanqa", "scan2cap"):
            order, off = official_per_item(key, rows)
            our = []
            missing = 0
            for row in rows:
                rec = record_for(key, row)
                if rec is None:
                    missing += 1
                    continue
                our.append(M.METRICS.evaluate(key, rec, row["pred_response"]))
            print(f"  --- per-sample vs official per-item arrays (missing records: {missing}) ---")
            for metric in ("CIDEr", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"):
                ref = [v * 100 for v in off[metric]]
                if key == "scan2cap" and metric == "CIDEr":
                    # the scan2cap baseline file was produced on a re-ordered
                    # question list, so align the WebUI corpus with the file's
                    # own ground truth before comparing
                    tmp = M.MetricEngine()
                    style = M.scan2cap_style
                    tmp._corpus_cache["scan2cap"] = [
                        (row["sample_id"], [style(r) for r in row["gt_response"]]) for row in rows
                    ]
                    gtmap = {row["sample_id"]: [style(r) for r in row["gt_response"]] for row in rows}
                    got = [
                        tmp._cider_single("scan2cap", row["sample_id"], style(row["pred_response"]), gtmap[row["sample_id"]]) * 100
                        for row in rows
                    ]
                else:
                    got = [r[metric] for r in our]
                n = min(len(ref), len(got))
                maxdiff = max(abs(ref[i] - got[i]) for i in range(n))
                worst[metric] = max(worst.get(metric, 0.0), maxdiff)
                flag = "OK " if maxdiff < 1e-3 else "!! "
                print(f"    {flag}{metric:<8} n={n:<6} max|diff|={maxdiff:.6f}   mean(ours)={sum(got)/n:.6f}   mean(official)={sum(ref)/n:.6f}")
        else:
            our = []
            for row in rows:
                rec = record_for(key, row)
                if rec is None:
                    continue
                our.append(M.METRICS.evaluate(key, rec, prediction_of(key, row)))
            metric_set = config.dataset_info(key)["metric_set"]
            if metric_set == "sqa3d":
                mean = sum(r["EM(acc)"] for r in our) / len(our) * 100
                print(f"    EM(acc) mean(per-sample)={mean:.10f}")
            elif metric_set == "scanrefer":
                for m in ("IoU@0.25", "IoU@0.5"):
                    mean = sum(r[m] for r in our) / len(our) * 100
                    print(f"    {m:<9} mean(per-sample)={mean:.10f}")
            elif metric_set == "multi3drefer":
                for m in ("F1@0.25", "F1@0.5"):
                    mean = sum(r[m] for r in our) / len(our)
                    print(f"    {m:<9} mean(per-sample)={mean:.10f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
