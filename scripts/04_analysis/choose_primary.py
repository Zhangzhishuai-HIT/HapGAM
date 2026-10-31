"""Pre-registered primary-model choice between HapGAM-A and HapGAM-E (docs/PREREGISTRATION.md).

Mean Pearson r on dev_region, averaged over folds within each primary trait and then over the four
primary traits; the higher variant wins, and HapGAM-E is chosen when the difference is < 0.002.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.data import PRIMARY_TRAITS  # noqa: E402
from hapgam.paths import RESULTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import collect  # noqa: E402

df = collect("dev_region")
df = df[df["trait"].isin(PRIMARY_TRAITS) & df["model"].isin(["hapgam", "hapgam_e"])]
counts = df.groupby(["model", "trait"]).size().unstack()
assert (counts == 6).all().all(), f"incomplete folds:\n{counts}"
per_trait = df.groupby(["model", "trait"])["pearson"].mean().unstack()
overall = per_trait.mean(axis=1)
diff = float(overall["hapgam_e"] - overall["hapgam"])
primary = "hapgam_e" if diff > -0.002 else "hapgam"
if abs(diff) < 0.002:
    primary = "hapgam_e"
res = {"per_trait": per_trait.round(4).to_dict(), "overall": overall.round(4).to_dict(), "diff_e_minus_a": diff,
       "primary": primary, "secondary": "hapgam" if primary == "hapgam_e" else "hapgam_e"}
(RESULTS / "tables").mkdir(parents=True, exist_ok=True)
(RESULTS / "tables" / "primary_model_choice.json").write_text(json.dumps(res, indent=1))
print(per_trait.round(4).to_string())
print(json.dumps({k: v for k, v in res.items() if k != "per_trait"}, indent=1))
