"""Local task runner for the Xiyou CPU server (no scheduler there).

Each line of the task list is "<model>\t<scenario>\t<trait>\t<fold>"; tasks run in list order with a
fixed number of concurrent slots, each limited to --threads CPU threads.  run_model.py skips folds
whose result already exists, so the runner can be restarted at any time.
usage: python jobs/xiyou/xiyou_runner.py --slots 8 --threads 5 jobs/lists/xiyou_tasks.tsv
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.path.expanduser("~/miniforge3/envs/hapgam/bin/python")


def run(line, threads):
    model, sc, trait, fold = line.rstrip("\n").split("\t")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fold)
    log = ROOT / "logs" / "xiyou" / f"{model}__{sc}__{trait}__{safe}.log"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "scripts/run_model.py", "--model", model, "--scenario", sc, "--traits", trait,
                              "--folds", fold, "--threads", str(threads)], cwd=ROOT, env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
    print(f"{time.strftime('%F %T')} rc={rc} {time.time() - t0:7.0f}s {model} {sc} {trait} {fold}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    (ROOT / "logs" / "xiyou").mkdir(parents=True, exist_ok=True)
    lines = [l for l in Path(a.tasks).read_text().splitlines() if l.strip()]
    print(f"{time.strftime('%F %T')} {len(lines)} tasks, {a.slots} slots x {a.threads} threads", flush=True)
    with ThreadPoolExecutor(a.slots) as ex:
        for l in lines:
            ex.submit(run, l, a.threads)
    print(f"{time.strftime('%F %T')} all tasks finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
