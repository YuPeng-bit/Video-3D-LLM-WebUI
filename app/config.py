"""Central configuration and dataset registry for the Video-3D-LLM WebUI.

The WebUI is a *separate* project: it never writes into the reference
upstream repository (which stays read-only). All it does is
import ``llava`` from there and reuse the official preprocessing / model code
so that every prediction is produced by exactly the official code path.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = os.environ.get("V3D_REPO",str(Path(__file__).resolve().parents[2]/'Video-3D-LLM'))
WEBUI_ROOT = os.environ.get("V3D_WEBUI",str(Path(__file__).resolve().parents[1]))
CKPT_NAME = os.environ.get("V3D_CKPT_NAME", "Video3D-LLM-LLaVA-Qwen-Uniform-32")
CKPT_PATH = os.environ.get("V3D_CKPT", os.path.join(REPO_ROOT, "ckpt", CKPT_NAME))

RESULTS_DIR = os.path.join(WEBUI_ROOT, "results", "webui")
THUMB_CACHE_DIR = os.path.join(WEBUI_ROOT, "results", "thumbcache")
LOG_DIR = os.path.join(WEBUI_ROOT, "logs")

for _d in (RESULTS_DIR, THUMB_CACHE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Official defaults (must stay identical to the upstream eval scripts)
# --------------------------------------------------------------------------
# scripts/3d/eval/model_*.py --extra-prompt default
DEFAULT_EXTRA_PROMPT = (
    "The video captures 3D spatial information of a scene. Please focus on the "
    "spatial relationships in the video and answer the following questions.\n"
)
DEFAULT_FRAME_SAMPLING_STRATEGY = os.environ.get("V3D_FRAME_STRATEGY", "uniform")
DEFAULT_MAX_FRAME_NUM = int(os.environ.get("V3D_MAX_FRAMES", "32"))
DEFAULT_FORCE_SAMPLE = True
DEFAULT_CONV_MODE = "qwen_1_5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = None
DEFAULT_NUM_BEAMS = 1
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_DECODE_STRATEGY = "center_crop"   # VideoProcessor.preprocess(strategy=...)

# ``overwrite_cfg`` as used by every eval script (``--overwrite_cfg true``)
OVERWRITE_CFG = {
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 151649,
}

# --------------------------------------------------------------------------
# Chain-of-thought ("thinking block") presets
# --------------------------------------------------------------------------
# The released checkpoint is a *Qwen2-7B* based LLaVA model: there is no
# ``enable_thinking`` switch and no `` thinking`` special token in the vocab
# (vocab_size 151643, added tokens are only <image>/<ground>/<coord> and the
# Qwen chat markers).  The only way to get a visible reasoning block out of it
# is therefore prompt-injected CoT (+ optionally forcing the first tokens).
#
# Every preset prepends an instruction to the official --extra-prompt and asks
# the model to separate reasoning from the final answer, so the WebUI can split
# the two and still score the answer with the official metrics.
COT_PRESETS = {
    "off": {
        "label": "关闭（= 官方默认，无思考块）",
        "instruction": "",
        "style": None,
        "split": None,
    },
    "brief": {
        "label": "轻量：先想一步再回答",
        "instruction": "Think about the 3D spatial layout of the scene step by step before answering.\n",
        "style": "plain",
        "split": False,
    },
    "structured_zh": {
        "label": "结构化：思考 / 答案（中文标记）",
        "instruction": (
            "First reason about the 3D spatial layout step by step, then answer.\n"
            "Output exactly in this format, with nothing before it:\n"
            "思考：<your step-by-step reasoning in 1-3 sentences>\n"
            "答案：<the final short answer>\n"
        ),
        "style": "zh",
        "split": True,
        "force_prefix": "思考：",
    },
    "structured_en": {
        "label": "Structured: Thinking / Answer (English markers)",
        "instruction": (
            "First reason about the 3D spatial layout step by step, then answer.\n"
            "Output exactly in this format, with nothing before it:\n"
            "Thinking: <your step-by-step reasoning in 1-3 sentences>\n"
            "Answer: <the final short answer>\n"
        ),
        "style": "en",
        "split": True,
        "force_prefix": "Thinking:",
    },
    "custom": {
        "label": "自定义（在下方编辑指令）",
        "instruction": "",
        "style": "custom",
        "split": True,
    },
}

COT_ORDER = ["off", "brief", "structured_zh", "structured_en", "custom"]

FRAME_SAMPLING_STRATEGIES = [
    "uniform",
    "mc-ratio90",
    "mc-ratio95",
]

# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------
# ``mode`` selects the official inference path:
#   generate  -> model.generate(...)                       (model_scanqa / model_sqa3d / model_scan2cap)
#   grounding -> model(..., use_object_proposals=True)     (model_scanrefer / model_multi3drefer)
#   caption   -> model.generate(...) with a <coord> box hint
#
# ``metric_set`` selects the official metric implementation used by
# llava/eval/eval_<dataset>.py
DATASETS = {
    "scanqa": {
        "key": "scanqa",
        "label": "ScanQA",
        "dataset_field": "scanQA",
        "question_file": "data/processed/scanqa_val_llava_style.json",
        "split": "val",
        "mode": "generate",
        "metric_set": "scanqa",
        "task_type": "question answering",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_scanqa.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scanqa.sh $CKPT uniform 32",
        "reference_metrics": {
            "CIDEr": 102.16229907535768,
            "BLEU-1": 47.101587072802296,
            "BLEU-2": 31.622544537462165,
            "BLEU-3": 22.593420680668856,
            "BLEU-4": 15.81619851121378,
            "METEOR": 19.876435066133315,
            "ROUGE-L": 49.076598864059285,
            "EM": 0.30117647058823527,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scanqa_baseline.txt",
    },
    "scan2cap": {
        "key": "scan2cap",
        "label": "Scan2Cap",
        "dataset_field": "scan2cap",
        "question_file": "data/processed/scan2cap_val_llava_style.json",
        "split": "val",
        "mode": "caption",
        "metric_set": "scan2cap",
        "task_type": "dense captioning",
        "expects_box_input": True,
        "official_eval_script": "llava/eval/eval_scan2cap.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scan2cap.sh $CKPT uniform 32",
        "reference_metrics": {
            "CIDEr": 83.83649561980529,
            "BLEU-4": 41.29272209370019,
            "METEOR": 28.821606148812823,
            "ROUGE-L": 62.305666961213355,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scan2cap_baseline.txt",
    },
    "scanrefer": {
        "key": "scanrefer",
        "label": "ScanRefer",
        "dataset_field": "scanrefer",
        "question_file": "data/processed/scanrefer_vg_val_llava_style.json",
        "split": "val",
        "mode": "grounding",
        "metric_set": "scanrefer",
        "task_type": "3D visual grounding",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_scanrefer.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scanrefer.sh $CKPT uniform 32",
        "reference_metrics": {
            "IoU@0.25": 58.182583087925956,
            "IoU@0.5": 51.75641564997896,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scanrefer_baseline.txt",
    },
    "multi3drefer": {
        "key": "multi3drefer",
        "label": "Multi3DRefer",
        "dataset_field": "multi3drefer",
        "question_file": "data/processed/multi3drefer_val_llava_style.json",
        "split": "val",
        "mode": "grounding",
        "metric_set": "multi3drefer",
        "task_type": "multi-object 3D visual grounding",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_multi3drefer.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_multi3drefer.sh $CKPT uniform 32",
        "reference_metrics": {
            "F1@0.25": 0.574089126896188,
            "F1@0.5": 0.5213496196772585,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/multi3drefer_baseline.txt (0-1 scale, threshold 0.4)",
    },
    "sqa3d": {
        "key": "sqa3d",
        "label": "SQA3D (test)",
        "dataset_field": "sqa3d",
        "question_file": "data/processed/sqa3d_test_llava_style.json",
        "split": "test",
        "mode": "generate",
        "metric_set": "sqa3d",
        "task_type": "situated question answering",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_sqa3d.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_sqa3d.sh $CKPT uniform 32",
        "reference_metrics": {
            "EM(acc)": 58.48252344416027,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/sqa3d_baseline.txt",
    },
}

DATASET_ORDER = ["scanqa", "scan2cap", "scanrefer", "multi3drefer", "sqa3d"]


def dataset_info(key: str) -> dict:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset: {key}. known: {list(DATASETS)}")
    return DATASETS[key]


def abs_question_file(key: str) -> str:
    return os.path.join(REPO_ROOT, dataset_info(key)["question_file"])
