"""Check SpectralREML against rrBLUP::mixed.solve (R) on a GRIN subset."""
import os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hapgam import data
from hapgam.reml import SpectralREML, center_genotypes, vanraden_kernel

RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript"   # Rscript of the hapgam_r environment
tmp = Path(__file__).parent / "tmp"
tmp.mkdir(exist_ok=True)

def test_matches_rrblup():
    rng = np.random.default_rng(1)
    part = next(data.scenario_partitions("dev_random"))
    td = data.build_trait_data("seed_weight_g", part[1], part[2])
    idx = rng.choice(len(td.train_rows), 1500, replace=False)
    snp = np.sort(rng.choice(data.grin_dosage().shape[1], 3000, replace=False))
    X = np.asarray(data.grin_dosage()[td.train_rows[idx]][:, snp], float)
    y = td.y_train[idx]
    ntr = 1200
    np.savetxt(tmp / "X.txt", X); np.savetxt(tmp / "y.txt", y)
    r = f"""
    library(rrBLUP); X <- as.matrix(read.table('{tmp}/X.txt')); y <- scan('{tmp}/y.txt', quiet=TRUE)
    tr <- 1:{ntr}; Z <- scale(X, center=colMeans(X[tr,]), scale=FALSE)
    K <- tcrossprod(Z)/ncol(Z)
    fit <- mixed.solve(y[tr], K=K[tr,tr], method='REML')
    pred <- as.numeric(fit$beta) + K[-tr,tr] %*% solve(K[tr,tr] + diag(fit$Ve/fit$Vu, {ntr})) %*% (y[tr]-as.numeric(fit$beta))
    write(c(fit$Vu, fit$Ve, fit$beta), '{tmp}/r_vc.txt', ncolumns=1)
    write(pred, '{tmp}/r_pred.txt', ncolumns=1)
    """
    subprocess.run([RSCRIPT, "-e", r], check=True, capture_output=True)
    vu, ve, b = np.loadtxt(tmp / "r_vc.txt")
    rpred = np.loadtxt(tmp / "r_pred.txt")
    Xt = torch.tensor(X)
    Ztr, Zte, _ = center_genotypes(Xt[:ntr], Xt[ntr:])
    K = vanraden_kernel(Ztr); Kte = vanraden_kernel(Zte, Ztr, scale=Ztr.shape[1])
    m = SpectralREML().fit(K, torch.tensor(y[:ntr]))
    pred = m.predict(Kte).numpy()
    print("delta py", m.delta, "R", ve / vu, "beta", float(m.beta[0]), b)
    print("pred corr", np.corrcoef(pred, rpred)[0, 1], "max abs diff", np.abs(pred - rpred).max())
    assert abs(np.log(m.delta) - np.log(ve / vu)) < 0.02
    assert np.abs(pred - rpred).max() < 1e-3 * np.std(rpred) + 1e-6

if __name__ == "__main__":
    test_matches_rrblup(); print("OK")
