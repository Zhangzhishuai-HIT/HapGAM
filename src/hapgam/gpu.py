"""GPU selection that never displaces other users' jobs.

A GPU is eligible only if its free memory exceeds the requested amount plus a safety
margin; the process then caps its own allocation with
``torch.cuda.set_per_process_memory_fraction`` so it cannot grow into memory that
other users' running jobs may need.
"""

from __future__ import annotations

import os
import subprocess


def gpu_table():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, tot, used, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "total": int(tot), "used": int(used), "util": int(util)})
    return rows


def pick_gpu(need_mb: int, margin_mb: int = 4096, exclude=()):
    """Return the index of the eligible GPU with the most free memory, or None."""
    best = None
    for r in gpu_table():
        if r["index"] in exclude:
            continue
        free = r["total"] - r["used"]
        if free >= need_mb + margin_mb and (best is None or free > best[1]):
            best = (r["index"], free)
    return None if best is None else best[0]


def setup_torch_device(cap_mb: int = 8192):
    """Bind to the GPU in CUDA_VISIBLE_DEVICES (set by the launcher) and cap memory use."""
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    dev = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(dev).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mb / total), dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    return dev


def limit_threads(n: int = 4):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
