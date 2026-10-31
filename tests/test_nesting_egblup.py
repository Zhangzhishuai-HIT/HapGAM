"""HapGAM-E restricted to M = 0 must reproduce the EGBLUP baseline exactly."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hapgam import api  # noqa: E402
from test_hapgam_synthetic import simulate  # noqa: E402


def test_hapgam_e_m0_equals_egblup():
    X, y, _ = simulate(n=1500)
    Xtr, Xte, ytr = X[:1200].astype(np.int8), X[1200:].astype(np.int8), y[:1200]
    dev = torch.device("cpu")
    p_e, _ = api.kernel_model("egblup", Xtr, ytr, Xte, dev)
    p_h, info = api.hapgam(Xtr, ytr, np.array(["a", "b", "c"] * 400), "random", Xte, dev,
                           api.schemes_for(np.zeros(X.shape[1], int), ("fixed8",)),
                           grid={"scheme": ["fixed8"], "top_m": [], "weight_decay": []}, backbone="AE")
    assert info["chosen"]["top_m"] == 0
    assert np.abs(p_e - p_h).max() < 1e-6 * np.std(p_e)


if __name__ == "__main__":
    test_hapgam_e_m0_equals_egblup()
    print("OK")
