"""Pick the GPU with the most free memory (shared cluster friendly)."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception as exc:
        print(f"[pick_gpu] nvidia-smi failed: {exc}", file=sys.stderr)
        print(0)
        return 0
    best, best_free = None, -1
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
        free = total - used
        print(f"[pick_gpu] gpu {idx}: used {used} MiB, free {free} MiB", file=sys.stderr)
        if free > best_free:
            best, best_free = idx, free
    if best is None:
        best = 0
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
