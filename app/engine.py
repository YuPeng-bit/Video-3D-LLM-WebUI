"""Video-3D-LLM inference engine used by the WebUI.

The engine intentionally reproduces ``llava/eval/model_<dataset>.py`` step by
step (same prompt assembly, same frame sampling, same pre/post-processing) so
that the "official default" configuration of the WebUI is identical to the
upstream evaluation scripts.  Everything the UI lets you change is an explicit
override of one of those official arguments.

Two mechanical differences are unavoidable and are documented where they
happen: the prompt pre-processor is imported from the upstream modules, and the
stopping criteria are wrapped in a ``StoppingCriteriaList`` (the container the
installed transformers version requires).
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import config, store

# The reference repository must be importable *before* ``llava`` is imported.
if config.REPO_ROOT not in sys.path:
    sys.path.insert(0, config.REPO_ROOT)
# ``llava.video_utils`` resolves several assets with relative paths.
os.chdir(config.REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import StoppingCriteriaList  # noqa: E402
from transformers.generation.logits_process import LogitsProcessorList  # noqa: E402
from transformers.generation.logits_process import LogitsProcessor  # noqa: E402

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from llava.video_utils import VideoProcessor, merge_video_dict  # noqa: E402


# --------------------------------------------------------------------------
# ``preprocess_qwen`` (verbatim copy of llava/eval/model_scanrefer.py)
# --------------------------------------------------------------------------
def _preprocess_qwen_fallback(sources, tokenizer: transformers.PreTrainedTokenizer,
                              has_image: bool = False, with_labels: bool = False,
                              system_message: str = "You are a helpful assistant."):
    """Verbatim copy of the upstream helper.

    ``model_scanqa.py`` returns only ``input_ids`` while ``model_scanrefer.py``
    returns ``(input_ids, targets)``; both share the same body.
    """
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_ids, targets = [], []
    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for sentence in source:
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1: -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    if with_labels:
        return input_ids, targets
    return input_ids


_PREPROCESS_CACHE: Dict[str, object] = {}


def _official_preprocess(module_name: str):
    """Reuse the upstream helper whenever it imports cleanly."""
    if module_name not in _PREPROCESS_CACHE:
        try:
            import importlib

            mod = importlib.import_module(f"llava.eval.{module_name}")
            _PREPROCESS_CACHE[module_name] = getattr(mod, "preprocess_qwen")
        except Exception:
            _PREPROCESS_CACHE[module_name] = None
    return _PREPROCESS_CACHE[module_name]


def preprocess_qwen(sources, tokenizer, has_image: bool = False, with_labels: bool = False):
    """Same variant the official script of that task uses."""
    fn = _official_preprocess("model_scanrefer" if with_labels else "model_scanqa")
    if fn is not None:
        return fn(sources, tokenizer, has_image=has_image)
    return _preprocess_qwen_fallback(sources, tokenizer, has_image=has_image, with_labels=with_labels)


# --------------------------------------------------------------------------
# Frame selection override
# --------------------------------------------------------------------------
class WebVideoProcessor(VideoProcessor):
    """``VideoProcessor`` whose frame list can be replaced per request.

    With ``frame_override = None`` the official behaviour (``np.linspace``
    uniform sampling over the embodiedscan frame list) is used unchanged.
    """

    frame_override: Optional[List[str]] = None

    def sample_frame_files(self, video_id: str, force_sample: bool = False, frames_upbound: int = 0):
        if self.frame_override:
            return list(self.frame_override)
        return super().sample_frame_files(video_id, force_sample=force_sample, frames_upbound=frames_upbound)


# --------------------------------------------------------------------------
# Output post-processing
# --------------------------------------------------------------------------
# Tagged reasoning blocks (used by Qwen3 / DeepSeek-R1 style models).
_THINK_TAG_PATTERNS = [
    re.compile(r"<\|?(?:redacted_)?think\|?>(.*?)<\|?/(?:redacted_)?think\|?>", re.S | re.I),
    re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.S | re.I),
]

# Marker style: "思考：... 答案：..." / "Thinking: ... Answer: ...".
# This is what the CoT presets ask for, because this Qwen2-based checkpoint has
# no native thinking tokens at all.
_THINK_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:思考|推理|分析|思路|理由|思考过程|Reasoning|Thinking|Thought)"
    r"\s*[:：]\s*(?P<think>.*?)"
    r"\s*(?:^|\n)\s*(?:答案|最终答案|结论|回答|Answer|Final\s+Answer)"
    r"\s*[:：]\s*(?P<answer>.*)$",
    re.S | re.I,
)

# Fallback: only a leading reasoning marker, no explicit answer marker.
_THINK_ONLY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:思考|推理|分析|思路|Reasoning|Thinking)\s*[:：]\s*(?P<think>.+)$",
    re.S | re.I,
)

# "The answer is ..." style trailing answer line.
_ANSWER_ONLY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:答案|最终答案|结论|Answer|Final\s+Answer)\s*[:：]\s*(?P<answer>.+)$",
    re.S | re.I,
)


def split_thinking(text: str):
    """Separate an explicit reasoning block from the final answer, if any.

    Supports both native tag styles and the marker style the WebUI CoT presets
    request (this checkpoint has no built-in thinking tokens).
    """
    if not text:
        return None, text
    for pat in _THINK_TAG_PATTERNS:
        m = pat.search(text)
        if m:
            thinking = m.group(1).strip()
            answer = (text[: m.start()] + text[m.end():]).strip()
            return thinking, answer

    m = _THINK_MARKER_PATTERN.search(text)
    if m:
        thinking = (m.group("think") or "").strip()
        answer = (m.group("answer") or "").strip()
        if thinking:
            return thinking, answer

    m = _THINK_ONLY_PATTERN.search(text)
    if m:
        thinking = (m.group("think") or "").strip()
        tail = _ANSWER_ONLY_PATTERN.search(thinking)
        if tail and tail.group("answer").strip():
            answer = tail.group("answer").strip()
            thinking = thinking[: tail.start()].strip()
            if thinking:
                return thinking, answer

    return None, text


class ForcePrefixLogitsProcessor(LogitsProcessor):
    """Force the first tokens of the generation to a fixed prefix.

    Used to *hard* enable a visible thinking block on a model that was never
    trained for it: the first tokens are forced to e.g. ``思考：`` so the model
    cannot answer directly and has to continue with reasoning.
    """

    def __init__(self, prefix_ids):
        self.prefix_ids = list(prefix_ids)
        self._step = 0

    def __call__(self, input_ids, scores):
        if self._step < len(self.prefix_ids):
            forced = self.prefix_ids[self._step]
            self._step += 1
            mask = torch.full_like(scores, float("-inf"))
            mask[:, forced] = 0.0
            return mask
        return scores


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.loaded = False
        self.loading = False
        self.load_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.model_name = get_model_name_from_path(config.CKPT_PATH)
        self._processors: Dict[str, WebVideoProcessor] = {}
        self._processor_lock = threading.RLock()

    # -- lifecycle ------------------------------------------------------
    def load(self) -> None:
        with self.lock:
            if self.loaded or self.loading:
                return
            self.loading = True
        t0 = time.time()
        try:
            disable_torch_init()
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                config.CKPT_PATH,
                None,
                self.model_name,
                overwrite_config=dict(config.OVERWRITE_CFG),
            )
            self.model.eval()
            self.loaded = True
        except Exception as exc:  # pragma: no cover
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.load_seconds = time.time() - t0
            self.loading = False

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    def unload(self) -> dict:
        """Release the checkpoint from GPU memory (CPU-side caches too).

        The next inference request simply loads it again (lazy loading), so
        this only trades ~10-140 s of reload time for the ~16 GB of VRAM.
        """
        with self.lock:
            if not self.loaded and self.model is None:
                return {"unloaded": False, "reason": "model was not loaded"}
            self.model = None
            self.tokenizer = None
            self.image_processor = None
            self.context_len = None
            self.loaded = False
            self.loading = False
            self.load_seconds = None
            with self._processor_lock:
                self._processors.clear()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        free = None
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            free = [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
        except Exception:
            pass
        return {"unloaded": True, "gpu_memory_used_mb": free}

    @property
    def device(self):
        return self.model.device

    def processor(self, strategy: str) -> WebVideoProcessor:
        with self._processor_lock:
            if strategy not in self._processors:
                self._processors[strategy] = WebVideoProcessor(
                    video_folder=os.path.join(config.REPO_ROOT, "data"),
                    annotation_dir=os.path.join(config.REPO_ROOT, "data", "embodiedscan"),
                    frame_sampling_strategy=strategy,
                )
            return self._processors[strategy]

    # -- planning (no GPU needed) ---------------------------------------
    def plan(
        self,
        key: str,
        index: int,
        *,
        extra_prompt: Optional[str] = None,
        question: Optional[str] = None,
        frame_indices: Optional[List[int]] = None,
        frame_strategy: Optional[str] = None,
        max_frames: Optional[int] = None,
        force_sample: Optional[bool] = None,
        decode_strategy: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        box_input: Optional[List[float]] = None,
        cot_mode: Optional[str] = None,
        cot_instruction: Optional[str] = None,
        cot_force_prefix: Optional[bool] = None,
        cot_force_prefix_text: Optional[str] = None,
    ) -> dict:
        info = config.dataset_info(key)
        record = store.STORE.get(key, index)
        scene = record.get("video")
        all_frames = store.STORE.scene_frames(scene)

        frame_strategy = frame_strategy or config.DEFAULT_FRAME_SAMPLING_STRATEGY
        max_frames = int(max_frames or config.DEFAULT_MAX_FRAME_NUM)
        force_sample = config.DEFAULT_FORCE_SAMPLE if force_sample is None else bool(force_sample)
        decode_strategy = decode_strategy or config.DEFAULT_DECODE_STRATEGY
        temperature = config.DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        top_p = config.DEFAULT_TOP_P if top_p is None else top_p
        num_beams = config.DEFAULT_NUM_BEAMS if num_beams is None else int(num_beams)
        max_new_tokens = config.DEFAULT_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
        extra_prompt = config.DEFAULT_EXTRA_PROMPT if extra_prompt is None else extra_prompt

        # ---- chain-of-thought ("thinking block") -------------------------
        cot_mode = cot_mode or "off"
        preset = config.COT_PRESETS.get(cot_mode) or config.COT_PRESETS["off"]
        if preset["style"] == "custom":
            cot_instruction = (cot_instruction or "").strip()
            if cot_instruction and not cot_instruction.endswith("\n"):
                cot_instruction += "\n"
        else:
            cot_instruction = preset["instruction"]
# NOTE: the upstream `--extra-prompt` is *recorded only* - llava/eval/
        # model_*.py computes `cur_prompt = extra_prompt + qs` but then feeds
        # `preprocess_qwen([line["conversations"][0], ...])`, i.e. the original
        # question without it.  The WebUI keeps that exact behaviour so the
        # default stays bit-identical to the official pipeline; the CoT
        # instruction is therefore injected into the *question* (which is
        # really sent to the model).
        effective_extra_prompt = extra_prompt
        force_text = preset.get("force_prefix")
        if preset["style"] == "custom":
            force_text = (cot_force_prefix_text or "").strip() or None
        mode_ok = info["mode"] in ("generate", "caption")
        if cot_force_prefix is None:
            cot_force_prefix = bool(force_text) and mode_ok
        else:
            cot_force_prefix = bool(cot_force_prefix) and mode_ok and bool(force_text)

        frame_override = bool(frame_indices)
        if frame_override:
            chosen = sorted({int(i) for i in frame_indices if 0 <= int(i) < len(all_frames)})
        else:
            chosen = store.default_frame_indices(len(all_frames), max_frames)
        frame_paths = [store.to_abs(all_frames[i]) for i in chosen]
        if not frame_paths:
            raise ValueError("no frames selected")

        conversation_value = (record.get("conversations") or [{}])[0].get("value", "")
        body = conversation_value
        if body.startswith(DEFAULT_IMAGE_TOKEN):
            body = body[len(DEFAULT_IMAGE_TOKEN):]
        body = body.lstrip()
        if question is not None:
            body = question
        body_effective = body
        if cot_instruction:
            instr = cot_instruction.strip()
            if instr:
                body_effective = f"{body}\n{instr}"
        final_value = f"{DEFAULT_IMAGE_TOKEN} {body_effective}".rstrip()

        qs = dict((record.get("conversations") or [{}])[0])
        qs["value"] = final_value
        cur_prompt = effective_extra_prompt + final_value

        use_im_start_end = bool(self.loaded and self.model.config.mm_use_im_start_end)
        if use_im_start_end:
            model_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + body_effective
        else:
            model_qs = DEFAULT_IMAGE_TOKEN + "\n" + body_effective
        conv = conv_templates[config.DEFAULT_CONV_MODE].copy()
        conv.append_message(conv.roles[0], model_qs)
        conv.append_message(conv.roles[1], None)

        return {
            "dataset": key,
            "label": info["label"],
            "index": index,
            "id": record.get("id"),
            "scene": scene,
            "question_type": (record.get("metadata") or {}).get("question_type"),
            "mode": info["mode"],
            "gt": store.STORE.gt_display(key, record),
            "prompt": cur_prompt,
            "model_prompt": conv.get_prompt(),
"extra_prompt": effective_extra_prompt,
            "extra_prompt_base": extra_prompt,
            "cot_mode": cot_mode,
            "cot_label": preset["label"],
            "cot_style": preset["style"],
            "cot_instruction": cot_instruction,
            "cot_force_prefix": cot_force_prefix,
            "cot_force_prefix_text": (force_text if cot_force_prefix else None),
            "question": body,
            "question_effective": body_effective,
            "cot_injected_in": ("question" if (cot_instruction and body_effective != body) else None),
            "conversation": qs,
            "frame_indices": chosen,
            "frame_paths": frame_paths,
            "frame_names": [os.path.basename(p) for p in frame_paths],
            "n_frames_available": len(all_frames),
            "frame_override": frame_override,
            "frame_strategy": frame_strategy,
            "max_frames": max_frames,
            "force_sample": force_sample,
            "decode_strategy": decode_strategy,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "box_input": box_input if box_input is not None else record.get("box_input"),
        }

    # -- main entry -----------------------------------------------------
    def run(self, key: str, index: int, *, record_only: bool = False, **kwargs) -> dict:
        plan = self.plan(key, index, **kwargs)
        if record_only:
            plan.update({"prediction": None, "raw_output": None, "thinking": None,
                         "answer_text": None, "metrics": None, "extra": None})
            return plan

        info = config.dataset_info(key)
        record = store.STORE.get(key, index)
        self.ensure_loaded()

        box_override = plan["box_input"]
        if info["expects_box_input"] and box_override:
            box_override = [float(v) for v in box_override][:3]
        else:
            box_override = None

        vp = self.processor(plan["frame_strategy"])
        with self.lock:
            try:
                vp.frame_override = plan["frame_paths"]
                t0 = time.time()
                if info["mode"] in ("generate", "caption"):
                    prediction, raw_output, extra = self._run_generate(
                        record, plan["conversation"],
                        vp=vp,
                        force_sample=plan["force_sample"],
                        max_frames=len(plan["frame_paths"]),
                        temperature=plan["temperature"],
                        top_p=plan["top_p"],
                        num_beams=plan["num_beams"],
                        max_new_tokens=plan["max_new_tokens"],
                        box_input=box_override,
                        force_prefix=plan["cot_force_prefix_text"],
                    )
                elif info["mode"] == "grounding":
                    prediction, raw_output, extra = self._run_grounding(
                        key, record,
                        vp=vp,
                        force_sample=plan["force_sample"],
                        max_frames=len(plan["frame_paths"]),
                    )
                else:
                    raise KeyError(info["mode"])
                latency = time.time() - t0
            finally:
                vp.frame_override = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        thinking, answer_text = split_thinking(raw_output if isinstance(raw_output, str) else "")
        if isinstance(raw_output, str):
            # When a reasoning block was produced, the official metrics are
            # computed on the *answer* part only; the full text stays visible.
            metrics_input = answer_text if thinking else raw_output
            metrics_subject = "answer-part" if thinking else "raw-output"
        else:
            metrics_input = prediction
            metrics_subject = "prediction"
        plan.update({
            "prediction": prediction,
            "raw_output": raw_output,
            "thinking": thinking,
            "answer_text": answer_text if isinstance(raw_output, str) else None,
            "metrics_input": metrics_input,
            "metrics_subject": metrics_subject,
            "latency_s": round(latency, 3),
            "extra": extra,
        })
        return plan

    # -- inference paths ------------------------------------------------
    def _build_video_dict(self, vp, record, force_sample, max_frames, box_input=None):
        video_dict = vp.process_3d_video(
            record.get("video"),
            self.image_processor,
            force_sample=force_sample,
            frames_upbound=max_frames,
        )
        if box_input is not None:
            video_dict["box_input"] = box_input
        video_dict = merge_video_dict([video_dict])
        image_tensors = video_dict.pop("images").half().to(self.device)
        for k in video_dict:
            video_dict[k] = video_dict[k].half().to(self.device)
        return image_tensors, video_dict

    def _run_generate(self, record, qs, *, vp, force_sample, max_frames,
                      temperature, top_p, num_beams, max_new_tokens, box_input=None, force_prefix=None):
        conv = conv_templates[config.DEFAULT_CONV_MODE].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = preprocess_qwen(
            [qs, {"from": "gpt", "value": None}], self.tokenizer, has_image=True
        ).to(self.device)

        image_tensors, video_dict = self._build_video_dict(
            vp, record, force_sample, max_frames, box_input=box_input
        )
        stopping_criteria = StoppingCriteriaList(
            [KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
        )

        logits_processor = LogitsProcessorList()
        forced_prefix_ids = None
        if force_prefix:
            forced_prefix_ids = self.tokenizer(force_prefix, add_special_tokens=False).input_ids
            if forced_prefix_ids:
                logits_processor.append(ForcePrefixLogitsProcessor(forced_prefix_ids))

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                modalities="video",
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                video_dict=video_dict,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        extra = {
            "n_frames": int(image_tensors.shape[1]),
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(output_ids.shape[1]),
            "box_input": box_input,
            "forced_prefix": force_prefix,
            "forced_prefix_ids": forced_prefix_ids,
        }
        return outputs, outputs, extra

    def _run_grounding(self, key, record, *, vp, force_sample, max_frames):
        conv = record.get("conversations") or []
        sources = [conv[0], conv[1]] if len(conv) > 1 else [conv[0], {"from": "gpt", "value": None}]
        input_ids, labels = preprocess_qwen(sources, self.tokenizer, has_image=True, with_labels=True)
        input_ids, labels = input_ids.to(self.device), labels.to(self.device)

        image_tensors, video_dict = self._build_video_dict(vp, record, force_sample, max_frames)
        objects = video_dict["objects"][0]

        with torch.inference_mode():
            _, scores = self.model(
                input_ids,
                images=image_tensors,
                modalities="video",
                video_dict=video_dict,
                labels=labels,
                use_object_proposals=True,
                box_labels=None,
            )

        extra = {
            "n_frames": int(image_tensors.shape[1]),
            "input_tokens": int(input_ids.shape[1]),
            "n_objects": int(objects.shape[0]),
            "scores": [float(s) for s in scores.tolist()],
            "objects": [[float(v) for v in o] for o in objects.tolist()],
        }

        if key == "scanrefer":
            try:
                _, pred_id = torch.max(scores, dim=0)
                pred_box = objects[pred_id].tolist()
            except Exception:
                _, pred_id = torch.max(scores[:-1], dim=0)  # drop the zero-target slot
                pred_box = objects[pred_id].tolist()
            extra["pred_object_index"] = int(pred_id)
            return [float(v) for v in pred_box], None, extra

        # multi3drefer: the official result file stores scores + objects; the
        # evaluation script applies the 0.4 cumulative-softmax rule.
        prediction = {"scores": extra["scores"], "objects": extra["objects"]}
        return prediction, None, extra


ENGINE = Engine()
