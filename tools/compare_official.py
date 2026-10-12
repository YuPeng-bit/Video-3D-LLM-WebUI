"""Compare WebUI default-config inference against the official result files.

The strongest possible faithfulness test: run the WebUI engine with *only*
official defaults and diff the prediction against the prediction the upstream
evaluation scripts wrote to ``results/.../baseline.jsonl``.

    python tools/compare_official.py --dataset scanqa --n 5
    python tools/compare_official.py --n 3            # every dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, engine as engine_mod, metrics as M, store  # noqa: E402

OFFICIAL = {
    "scanqa": "results/gt_structure_full/scanqa/baseline.jsonl",
    "scan2cap": "results/gt_structure_full/scan2cap/baseline.jsonl",
    "scanrefer": "results/gt_structure_full/scanrefer/baseline.jsonl",
    "multi3drefer": "results/gt_structure_full/multi3drefer/baseline.jsonl",
    "sqa3d": "results/gt_structure_full/sqa3d/baseline.jsonl",
}


def load_official(key: str):
    path = os.path.join(config.REPO_ROOT, OFFICIAL[key])
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                item = json.loads(line)
                out[item["sample_id"]] = item
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--index", type=int, default=0, help="first question index to test")
    ap.add_argument("--indices", type=str, default=None, help="explicit comma separated indices")
    args = ap.parse_args()

    keys = [args.dataset] if args.dataset else config.DATASET_ORDER
    engine_mod.ENGINE.ensure_loaded()
    print(f"[compare] model {engine_mod.ENGINE.model_name} on {engine_mod.ENGINE.device}")

    total_match = 0
    total = 0
    for key in keys:
        official = load_official(key)
        if not official:
            print(f"### {key}: no official result file, skipped")
            continue
        indices = ([int(x) for x in args.indices.split(",")] if args.indices
                   else list(range(args.index, args.index + args.n)))
        print("=" * 80)
        print(f"### {key}  (official predictions available: {len(official)})")
        for idx in indices:
            record = store.STORE.get(key, idx)
            sid = record.get("id")
            res = engine_mod.ENGINE.run(key, idx)
            ref = official.get(sid)
            total += 1
            if ref is None:
                print(f"  [#{idx} {sid}] no official entry -> skip comparison")
                continue
            if key == "multi3drefer":
                ours, theirs = res["prediction"], ref
                same = _sel(ours, 0.4) == _sel(theirs, 0.4)
            else:
                ours, theirs = res["prediction"], ref["pred_response"]
                same = ours == theirs
            total_match += int(same)
            print(f"  [#{idx} {sid}] {'MATCH' if same else 'DIFF'}")
            print(f"      ours   : {json.dumps(ours, ensure_ascii=False)[:220]}")
            print(f"      official: {json.dumps(theirs, ensure_ascii=False)[:220]}")
            if not same:
                m = M.METRICS.evaluate(key, record, ours)
                print(f"      our metrics: {json.dumps({k: v for k, v in m.items() if not k.startswith('_')})}")
    print("=" * 80)
    print(f"[compare] identical predictions: {total_match}/{total}")
    return 0 if total and total_match == total else 1


def _sel(pred, thr):
    _, _, boxes = M._select_objects(pred, thr)
    return [[round(v, 4) for v in b] for b in boxes]


if __name__ == "__main__":
    sys.exit(main())
