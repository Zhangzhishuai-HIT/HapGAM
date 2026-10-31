"""Gradient-boosted tree baselines (LightGBM / XGBoost) tuned with Optuna on inner CV."""

from __future__ import annotations

import time

import numpy as np

from .metrics import pearson


def _progress(tag):
    t0 = time.time()

    def cb(study, trial):
        val = "pruned" if trial.value is None else f"{trial.value:.4f}"
        print(f"[{tag}] trial {trial.number} value={val} best={study.best_value:.4f} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    return cb


def _lgb_params(trial, n_threads, seed):
    return {
        "objective": "regression", "verbosity": -1, "num_threads": n_threads, "seed": seed,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.02, 0.5, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def tune_lightgbm(X, y, folds, n_trials=25, n_threads=16, seed=0, max_rounds=2000, timeout=None):
    """Return (best_params, best_rounds, study).  ``folds`` = list of validation index arrays."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(y)
    bin_params = {"max_bin": 7, "min_data_in_bin": 3, "verbosity": -1, "num_threads": n_threads,
                  "feature_pre_filter": False}
    full = lgb.Dataset(X, y, params=bin_params, free_raw_data=False).construct()
    sets = []
    for va in folds:
        tr = np.setdiff1d(np.arange(n), va)
        sets.append((full.subset(tr).construct(), full.subset(va).construct(), va))

    def objective(trial):
        p = _lgb_params(trial, n_threads, seed)
        scores, iters = [], []
        for dtr, dva, va in sets:
            bst = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = bst.predict(X[va], num_iteration=bst.best_iteration)
            scores.append(pearson(y[va], pred))
            iters.append(bst.best_iteration)
            trial.report(float(np.mean(scores)), len(scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("fold_r", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[_progress("lgb")])
    best = study.best_trial
    rounds = int(np.mean(best.user_attrs["iters"]) * 1.1) + 1
    params = {**_lgb_params(optuna.trial.FixedTrial(best.params), n_threads, seed)}
    return params, rounds, study


def fit_lightgbm(X, y, params, rounds):
    import lightgbm as lgb

    ds = lgb.Dataset(X, y, params={"max_bin": 7, "min_data_in_bin": 3, "verbosity": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=rounds)


def _xgb_params(trial, seed, device):
    return {
        "objective": "reg:squarederror", "tree_method": "hist", "device": device, "seed": seed,
        "max_bin": 8, "verbosity": 0,
        "eta": trial.suggest_float("eta", 0.02, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.02, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }


def tune_xgboost(X, y, folds, n_trials=25, seed=0, device="cpu", nthread=16, max_rounds=2000):
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(y)
    sets = []
    for va in folds:
        tr = np.setdiff1d(np.arange(n), va)
        sets.append((xgb.QuantileDMatrix(X[tr], y[tr], max_bin=8, nthread=nthread),
                     xgb.DMatrix(X[va], y[va], nthread=nthread), va))

    def objective(trial):
        p = _xgb_params(trial, seed, device)
        p["nthread"] = nthread
        scores, iters = [], []
        for dtr, dva, va in sets:
            bst = xgb.train(p, dtr, num_boost_round=max_rounds, evals=[(dva, "va")],
                            early_stopping_rounds=100, verbose_eval=False)
            pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            scores.append(pearson(y[va], pred))
            iters.append(bst.best_iteration + 1)
            trial.report(float(np.mean(scores)), len(scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("fold_r", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials, callbacks=[_progress("xgb")])
    best = study.best_trial
    rounds = int(np.mean(best.user_attrs["iters"]) * 1.1) + 1
    params = _xgb_params(optuna.trial.FixedTrial(best.params), seed, device)
    params["nthread"] = nthread
    return params, rounds, study


def fit_xgboost(X, y, params, rounds):
    import xgboost as xgb

    return xgb.train(params, xgb.QuantileDMatrix(X, y, max_bin=8), num_boost_round=rounds)
