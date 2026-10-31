"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p) -> float:
    if len(y) < 3:
        return np.nan
    return float(stats.spearmanr(y, p).correlation)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def top_coincidence(y, p, frac: float = 0.2) -> float:
    """Share of the true top-`frac` set recovered by the predicted top-`frac` set."""
    n = len(y)
    k = max(1, int(round(frac * n)))
    top_y = set(np.argsort(-np.asarray(y))[:k])
    top_p = set(np.argsort(-np.asarray(p))[:k])
    return len(top_y & top_p) / k


def within_batch_r(records: pd.DataFrame, pred_by_row: dict, min_n: int = 20) -> float:
    """Mean over batches (>= min_n test records) of Pearson r between record value and prediction."""
    rs = []
    for _, g in records.groupby("batch"):
        if len(g) < min_n:
            continue
        p = g["row"].map(pred_by_row).to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(g["c"].to_numpy(float))
        if ok.sum() >= min_n:
            rs.append(pearson(g["c"].to_numpy(float)[ok], p[ok]))
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan


def summary(y, p, records: pd.DataFrame | None = None, rows=None) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    pred_by_row = dict(zip(rows, p)) if rows is not None else None
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), "pearson": pearson(y, p), "spearman": spearman(y, p),
           "rmse": rmse(y, p), "top20": top_coincidence(y, p)}
    if records is not None and rows is not None:
        out["within_batch_r"] = within_batch_r(records, pred_by_row)
    return out


def auc(y_bin, score) -> float:
    y_bin = np.asarray(y_bin).astype(bool)
    if y_bin.all() or (~y_bin).all():
        return np.nan
    ranks = stats.rankdata(score)
    n1 = y_bin.sum()
    n0 = len(y_bin) - n1
    return float((ranks[y_bin].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_group_r(y, p, groups, min_n: int = 20) -> float:
    """Mean Pearson r within groups (families, MG tests) having >= min_n finite pairs."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    groups = np.asarray(groups)
    rs = []
    for g in pd.unique(groups):
        m = (groups == g) & np.isfinite(y) & np.isfinite(p)
        if m.sum() >= min_n:
            r = pearson(y[m], p[m])
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else np.nan
