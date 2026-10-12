"""End-to-end smoke test: run one official inference per dataset.

    python tools/smoke_test.py [scanqa scan2cap ...] [--index N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, engine as engine_mod, metrics as M, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    keys = args.datasets or config.DATASET_ORDER
    eng = engine_mod.ENGINE
    print(f"[smoke] ckpt = {config.CKPT_PATH}")
    print(f"[smoke] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    eng.ensure_loaded()
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s on {eng.device}")

    failures = 0
    for key in keys:
        rec = store.STORE.get(key, args.index)
        print("\n" + "=" * 78)
        print(f"### {key}  index={args.index}  id={rec.get('id')}  scene={rec.get('video')}")
        try:
            res = eng.run(key, args.index)
            print(f"  frames          : {res['frame_indices']}")
            print(f"  prediction      : {json.dumps(res['prediction'], ensure_ascii=False)[:400]}")
            print(f"  raw output      : {json.dumps(res['raw_output'], ensure_ascii=False)[:400]}")
            print(f"  thinking        : {res['thinking']}")
            print(f"  latency         : {res['latency_s']}s")
            if not args.no_metrics:
                m = M.METRICS.evaluate(key, rec, res["prediction"])
                print("  official metrics: " + json.dumps({k: v for k, v in m.items() if not k.startswith('_')}, ensure_ascii=False))
        except Exception as exc:
            failures += 1
            import traceback

            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("\n" + "=" * 78)
    print(f"[smoke] done, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
