"""Probe the Chain-of-Thought ("thinking block") presets.

    CUDA_VISIBLE_DEVICES=1 python tools/cot_probe.py --dataset scanqa --index 0

For each preset it prints the raw output, the split thinking / answer parts and
the official metrics computed on the answer part.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, engine as engine_mod, metrics as M, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scanqa")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--modes", default="off,brief,structured_zh,structured_en")
    ap.add_argument("--no-force", action="store_true", help="disable prefix forcing")
    ap.add_argument("--indices", default=None, help="comma separated indices, overrides --index")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    indices = [int(x) for x in args.indices.split(",")] if args.indices else [args.index]

    eng = engine_mod.ENGINE
    print(f"[cot] ckpt = {config.CKPT_PATH}")
    print(f"[cot] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    eng.ensure_loaded()
    print(f"[cot] model loaded on {eng.device}\n")

    for idx in indices:
        rec = store.STORE.get(args.dataset, idx)
        print("=" * 90)
        print(f"### {args.dataset} #{idx}  id={rec.get('id')}  scene={rec.get('video')}")
        print(f"    question : {((rec.get('conversations') or [{}])[0].get('value') or '')[:150]}")
        print(f"    GT       : {json.dumps(store.STORE.gt_display(args.dataset, rec), ensure_ascii=False)[:200]}")
        for mode in modes:
            print("-" * 90)
            print(f"  [{mode}] {config.COT_PRESETS[mode]['label']}")
            try:
                res = eng.run(
                    args.dataset, idx,
                    cot_mode=mode,
                    cot_force_prefix=(False if args.no_force else None),
                )
            except Exception as exc:
                import traceback

                print(f"    [FAIL] {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            print(f"    force_prefix   : {res['cot_force_prefix']} {res['cot_force_prefix_text'] or ''}")
            print(f"    raw output     : {json.dumps(res['raw_output'], ensure_ascii=False)}")
            print(f"    thinking       : {json.dumps(res['thinking'], ensure_ascii=False)}")
            print(f"    answer part    : {json.dumps(res['answer_text'], ensure_ascii=False)}")
            print(f"    metrics on     : {res['metrics_subject']}")
            m = M.METRICS.evaluate(args.dataset, rec, res["metrics_input"])
            shown = {k: v for k, v in m.items() if not k.startswith("_")}
            print(f"    official metric: {json.dumps(shown, ensure_ascii=False)}")
            print(f"    latency        : {res['latency_s']}s  frames={len(res['frame_indices'])}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
