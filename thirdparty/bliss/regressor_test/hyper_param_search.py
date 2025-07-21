#!/usr/bin/env python3
"""
Grid-search hyper-parameters for Bliss regressors, now parallelised.

Usage example:
  # 8-core parallel run, 2-fold CV
  python hyper_param_search.py \
      --data_dir thirdparty/bliss/regressor_test/datasets/femnist \
      --model xgboost --regressor g \
      --n_jobs 8 --cv_splits 2
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore


def load_latest_csv(data_dir: Path, regressor: str) -> pd.DataFrame:
    csvs = sorted(data_dir.glob(f"{regressor}_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No {regressor}_*.csv in {data_dir}")
    logging.info("Dataset: %s", csvs[-1].name)
    return pd.read_csv(csvs[-1])


def eval_cfg(model_name: str,
             params: dict,
             X: np.ndarray,
             y: np.ndarray,
             rounds: np.ndarray,
             cv: int,
             rng_seed: int = 42) -> float:
    """Return unweighted mean RMSE across rounds, with early stopping."""
    is_xgb = (model_name == "xgboost")
    if is_xgb:
        from xgboost import XGBRegressor
    else:
        from sklearn.neural_network import MLPRegressor

    rmses: list[float] = []
    for r in np.unique(rounds):
        mask = (rounds == r)
        X_r, y_r = X[mask], y[mask]
        if len(X_r) < cv:
            continue

        kf = KFold(n_splits=cv, shuffle=True, random_state=rng_seed)
        se: list[float] = []

        for tr, va in kf.split(X_r):
            if is_xgb:
                model = XGBRegressor(
                    **params,
                    early_stopping_rounds=10
                )
                model.fit(
                    X_r[tr], y_r[tr],
                    eval_set=[(X_r[va], y_r[va])],
                    verbose=False
                )
            else:
                es_kwargs = dict(
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1
                )
                model = MLPRegressor(
                    **params,
                    random_state=rng_seed,
                    **es_kwargs
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    model.fit(X_r[tr], y_r[tr])
            y_hat = model.predict(X_r[va])
            se.append(mean_squared_error(y_r[va], y_hat))

        rmses.append(np.sqrt(np.mean(se)))

    return float(np.mean(rmses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="…/regressor_test/datasets/<job_name>")
    parser.add_argument("--model", required=True,
                        choices=["xgboost", "mlp"])
    parser.add_argument("--regressor", required=True,
                        choices=["g", "h"])
    parser.add_argument("--cv_splits", type=int, default=2,
                        help="Number of CV folds (default=2 for speed)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel configs to evaluate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

    df = load_latest_csv(Path(args.data_dir), args.regressor)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(np.float32)
    y = df["target_utility"].to_numpy(np.float32)
    rounds = df["round"].to_numpy(np.int32)

    if args.model == "xgboost":
        base_gpu = dict(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            gpu_id=0,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0
        )
        grid = {
            "n_estimators":     [200, 500],
            "learning_rate":    [0.02, 0.05, 0.1],
            "max_depth":        [3, 4, 6, 8],
            "subsample":        [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        }
        cfgs = [ {**base_gpu, **g} for g in ParameterGrid(grid) ]
    else:
        grid = {
            "hidden_layer_sizes": [(64,), (128,), (32,)],
            "learning_rate_init": [1e-3, 3e-4, 1e-4],
            "batch_size":         [32, 64],
            "alpha":              [1e-4, 1e-3],
        }
        cfgs = list(ParameterGrid(grid))

    n_cfg = len(cfgs)
    logging.info("Grid size: %d configurations", n_cfg)

    results: list[tuple[float, dict]] = []
    t0 = perf_counter()

    if args.n_jobs > 1:
        # parallel evaluation of configs
        executor = ProcessPoolExecutor(max_workers=args.n_jobs)
        futures = {
            executor.submit(eval_cfg,
                            args.model, cfg, X, y, rounds, args.cv_splits): cfg
            for cfg in cfgs
        }

        it = as_completed(futures)
        if tqdm:
            it = tqdm(it, total=n_cfg, desc="configs", unit="cfg")

        for fut in it:
            score = fut.result()
            cfg = futures[fut]
            results.append((score, cfg))
    else:
        # sequential (with progress bar)
        iterator = cfgs if not tqdm else tqdm(cfgs, desc="configs", unit="cfg")
        for cfg in iterator:
            score = eval_cfg(args.model, cfg, X, y, rounds, args.cv_splits)
            results.append((score, cfg))

    results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = results[0]
    elapsed = perf_counter() - t0

    print("\n========== BEST ==========")
    print(f"Model       : {args.model}")
    print(f"Regressor   : {args.regressor}")
    print(f"CV splits   : {args.cv_splits}")
    print(f"n_jobs      : {args.n_jobs}")
    print(f"Mean RMSE   : {best_rmse:.5f}")
    print(f"Hyper-params: {best_cfg}")
    print(f"Elapsed     : {elapsed/60:.1f} min")
    print("==========================")

    out = Path(args.data_dir) / f"best_{args.model}_{args.regressor}.json"
    with out.open("w") as f:
        json.dump([{"rmse": s, **c} for s, c in results], f, indent=2)
    logging.info("Full ranking written to %s", out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)