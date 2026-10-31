"""Do held-out SoyNAM families carry block haplotypes that the training lines lack?

For each SoyNAM fold in which HapGAM selected blocks, the share of test lines whose haplotype at a selected
block does not occur among the training lines, averaged over blocks with weights proportional to the
variance of the block contribution, and related (Spearman) to the gain of HapGAM over the kernel model
fitted alongside it (r_gblup_same_fit).  Reads the stored fold outputs and the frozen splits only.

usage: soynam_unseen_haplotypes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import scenarios  # noqa: E402
from hapgam.paths import RESULTS, RUNS  # noqa: E402

SCEN = ["soynam_lofo", "soynam_random"]
TRAITS = ["height_cm", "maturity_doy", "flowering_doy"]


def main():
    rows = []
    for sc in SCEN:
        for t in TRAITS:
            for f in scenarios.folds(sc, t):
                p = RUNS / sc / "hapgam_e" / t / f
                try:
                    z = np.load(f"{p}.parts.npz", allow_pickle=True)
                    j = json.loads(Path(f"{p}.json").read_text())
                except FileNotFoundError:
                    continue
                gain = j["pearson"] - j["r_gblup_same_fit"]
                sel = z["selected"]
                if len(sel) == 0:
                    rows.append((sc, t, f, 0, np.nan, gain))
                    continue
                task = scenarios.get_task(sc, t, f)
                Xtr, Xte = np.asarray(task.Xtr), np.asarray(task.Xte)
                imp = z["block_contrib"].var(0)
                w = imp / imp.sum()
                unseen = []
                for b in sel:
                    mem = z["scheme_idx"][b]
                    mem = mem[mem >= 0]
                    tr = {tuple(x) for x in np.rint(Xtr[:, mem]).astype(int)}
                    te = [tuple(x) for x in np.rint(Xte[:, mem]).astype(int)]
                    unseen.append(np.mean([h not in tr for h in te]))
                rows.append((sc, t, f, len(sel), float(np.dot(w, unseen)), gain))
    df = pd.DataFrame(rows, columns=["scenario", "trait", "fold", "M", "unseen_hap_share_w", "gain_vs_kernel"])
    df.to_csv(RESULTS / "tables" / "soynam_unseen_haplotypes.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))
    d = df.dropna()
    print("Spearman (folds with blocks):", stats.spearmanr(d.unseen_hap_share_w, d.gain_vs_kernel))


if __name__ == "__main__":
    main()
