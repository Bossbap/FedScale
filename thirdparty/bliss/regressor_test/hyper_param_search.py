#!/usr/bin/env python3
"""
Grid-search hyper-parameters for Bliss regressors (sequential, global CV).

Usage example:
  python thirdparty/bliss/hyper_param_search.py \
      --data_dir thirdparty/bliss/regressor_test/datasets/openimage_hp_collection \
      --model xgboost --regressor g \
      --n_jobs 1 --cv_splits 2 \
      --round_stride 2 --max_round 400
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
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

EARLY_STOP_ROUNDS = 20


def resolve_xgb_device(preferred: str | None) -> tuple[str, str, str]:
    """
    Return a tuple (device, tree_method, predictor) compatible with XGBoost ≥3.1.

    • If `preferred` is provided, use it verbatim (with CPU-specific defaults).
    """
    def _defaults(device_name: str) -> tuple[str, str, str]:
        is_cpu = device_name.lower() == "cpu"
        tree_method = "hist"
        predictor = "auto" if is_cpu else "gpu_predictor"
        return device_name, tree_method, predictor

    if preferred:
        return _defaults(preferred)

    # Fallback to CPU if nothing is specified
    return _defaults("cpu")


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
             rng_seed: int = 42,
             feature_names: list[str] | None = None) -> float:
    """
    Evaluate a single hyper-parameter config.

    - Perform global KFold CV over all samples.
    - Collect out-of-fold predictions.
    - Compute RMSE per round using these predictions.
    - Return the mean RMSE across rounds (unweighted).
    """
    model_name = model_name.lower()
    is_xgb = (model_name == "xgboost")
    is_lgbm = (model_name == "lightgbm")
    is_cat = (model_name == "catboost")

    if is_xgb:
        from xgboost import XGBRegressor
    elif is_lgbm:
        from lightgbm import LGBMRegressor, early_stopping as lgb_early_stop
    elif is_cat:
        from catboost import CatBoostRegressor
    else:
        from sklearn.neural_network import MLPRegressor

    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples, dtype=np.float32)

    kf = KFold(n_splits=cv, shuffle=True, random_state=rng_seed)

    for tr, va in kf.split(X):
        if is_xgb:
            model = XGBRegressor(**params)
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                verbose=False,
            )
            X_val_for_pred = X[va]

        elif is_lgbm:
            if feature_names is not None:
                X_tr = pd.DataFrame(X[tr], columns=feature_names)
                X_va = pd.DataFrame(X[va], columns=feature_names)
            else:
                X_tr = X[tr]
                X_va = X[va]
            model = LGBMRegressor(**params, random_state=rng_seed)
            model.fit(
                X_tr,
                y[tr],
                eval_set=[(X_va, y[va])],
                eval_metric="rmse",
                callbacks=[lgb_early_stop(
                    stopping_rounds=EARLY_STOP_ROUNDS,
                    verbose=False
                )],
            )
            X_val_for_pred = X_va

        elif is_cat:
            model = CatBoostRegressor(
                **params,
                random_seed=rng_seed,
                verbose=False
            )
            model.fit(
                X[tr],
                y[tr],
                eval_set=(X[va], y[va]),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose=False,
            )
            X_val_for_pred = X[va]

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
                model.fit(X[tr], y[tr])
            X_val_for_pred = X[va]

        y_hat = model.predict(X_val_for_pred)
        y_pred[va] = y_hat.astype(np.float32)

    # Per-round RMSE using out-of-fold predictions
    # Per-round RMSE using out-of-fold predictions
    rmses: list[float] = []
    for r in np.unique(rounds):
        mask = (rounds == r)
        if mask.sum() < cv:
            continue
        mse_r = mean_squared_error(y[mask], y_pred[mask])
        rmse_r = float(np.sqrt(mse_r))
        rmses.append(rmse_r)


    if not rmses:
        return float("inf")
    return float(np.mean(rmses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="…/regressor_test/datasets/<job_name>")
    parser.add_argument("--model", required=True,
                        choices=["xgboost", "lightgbm", "catboost", "mlp"])
    parser.add_argument("--regressor", required=True,
                        choices=["g", "h"])
    parser.add_argument("--cv_splits", type=int, default=2,
                        help="Number of CV folds (default=2 for speed)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Unused now (sequential only, kept for CLI compatibility)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="XGBoost device string (e.g. cuda:0 or cpu). Defaults to cpu.",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Optional cap on number of configs to evaluate (sampled from the grid)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path("thirdparty/bliss/regressor_test/hp_configs")),
        help="Directory to store results. Results saved under <save_dir>/<job_name>/",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Optional job_name; defaults to the last component of data_dir",
    )
    parser.add_argument(
        "--round_stride",
        type=int,
        default=1,
        help="Use only every k-th round (1 = use all rounds).",
    )
    parser.add_argument(
        "--max_round",
        type=int,
        default=None,
        help="If set, ignore samples from rounds > max_round.",
    )
    args = parser.parse_args()

    if args.round_stride < 1:
        raise ValueError(f"round_stride must be >= 1, got {args.round_stride}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

    df = load_latest_csv(Path(args.data_dir), args.regressor)

    # --------- round filtering: max_round then stride ----------
    if args.max_round is not None:
        before = df.shape[0]
        df = df[df["round"] <= args.max_round]
        logging.info(
            "Applied max_round=%d: kept %d / %d rows",
            args.max_round, df.shape[0], before
        )

    if args.round_stride > 1:
        unique_rounds = np.sort(df["round"].unique())
        selected_rounds = unique_rounds[::args.round_stride]
        before = df.shape[0]
        df = df[df["round"].isin(selected_rounds)]
        logging.info(
            "Applied round_stride=%d: kept %d / %d rows, %d rounds -> %d rounds",
            args.round_stride,
            df.shape[0],
            before,
            len(unique_rounds),
            len(selected_rounds),
        )
    # -----------------------------------------------------------

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(np.float32)
    y = df["target_utility"].to_numpy(np.float32)
    rounds = df["round"].to_numpy(np.int32)

    model = args.model.lower()

    if model == "xgboost":
        device, tree_method, predictor = resolve_xgb_device(args.device)
        logging.info("XGBoost device: %s (%s/%s)", device, tree_method, predictor)
        base_gpu = dict(
            tree_method=tree_method,
            predictor=predictor,
            device=device,
            objective="reg:pseudohubererror",
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbosity=0,
        )
        grid = {
            "n_estimators":      [200],
            "learning_rate":     [0.04, 0.1],
            "max_depth":         [3, 5, 8, 10],
            "subsample":         [0.7, 0.9, 1.0],
            "colsample_bytree":  [0.7, 0.9, 1.0],
            "min_child_weight":  [1, 3, 6],
            "reg_alpha":         [0, 0.1, 1],
            "reg_lambda":        [0.0, 3.0, 10.0],
            "huber_slope":       [1000, 2000, 3000],
        }
        cfgs_full = [{**base_gpu, **g} for g in ParameterGrid(grid)]

    elif model == "lightgbm":
        base_lgbm = dict(
            boosting_type="gbdt",
            objective="huber",
            n_estimators=200,
            verbosity=-1,
        )
        grid = {
            "num_leaves":        [31, 63, 127, 161],
            "learning_rate":     [0.02, 0.05, 0.1],
            "min_child_samples": [3, 8, 15],
            "colsample_bytree":  [0.7, 0.9, 1.0],
            "subsample":         [0.7, 0.9, 1.0],
            "subsample_freq":    [1],
            "reg_alpha":         [0.0],
            "reg_lambda":        [0.0, 1.0],
            "alpha":             [1000, 2000, 3000],  # Huber delta
        }
        cfgs_full = [{**base_lgbm, **g} for g in ParameterGrid(grid)]

    elif model == "catboost":
        grid = {
            "iterations":         [500],
            "learning_rate":      [0.02, 0.05, 0.1],
            "depth":              [3, 7, 10],
            "l2_leaf_reg":        [3, 10, 30],
            "bagging_temperature":[0, 0.5, 1, 2],
            "loss_function":      [f"Huber:delta={d}" for d in (1000, 2000, 3000)],
            "subsample":          [0.7, 1.0],
            "rsm": [0.7, 0.9, 1.0],
        }
        cfgs_full = list(ParameterGrid(grid))

    else:
        grid = {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (64, 64), (128, 64), (256, 128),
                (128, 128, 64),
            ],
            "activation":         ["relu", "tanh"],
            "learning_rate_init": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
            "batch_size":         [32, 64, 128, 256],
            "alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
            "solver":             ["adam"],
        }
        cfgs_full = list(ParameterGrid(grid))

    # Optionally subsample grid
    if args.max_configs is not None and args.max_configs < len(cfgs_full):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cfgs_full), size=args.max_configs, replace=False)
        cfgs = [cfgs_full[i] for i in idx]
    else:
        cfgs = cfgs_full

    # Resolve output directory under hp_configs/<job_name>
    job_name = args.job_name or Path(args.data_dir).name
    save_root = Path(args.save_dir) / job_name
    save_root.mkdir(parents=True, exist_ok=True)

    n_cfg = len(cfgs)
    logging.info("Grid size: %d configurations%s", n_cfg,
                 " (sampled)" if args.max_configs else "")

    results: list[tuple[float, dict]] = []
    t0 = perf_counter()

    # Sequential evaluation
    iterator = cfgs if not tqdm else tqdm(cfgs, desc="configs", unit="cfg")
    for cfg in iterator:
        score = eval_cfg(
            args.model,
            cfg,
            X,
            y,
            rounds,
            args.cv_splits,
            42,
            feat_cols,
        )
        results.append((score, cfg))

    results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = results[0]
    elapsed = perf_counter() - t0

    print("\n========== BEST ==========")
    print(f"Model       : {args.model}")
    print(f"Regressor   : {args.regressor}")
    print(f"CV splits   : {args.cv_splits}")
    print(f"n_jobs      : {args.n_jobs}  (sequential evaluation)")
    print(f"round_stride: {args.round_stride}")
    print(f"max_round   : {args.max_round}")
    print(f"Mean RMSE   : {best_rmse:.5f}")
    print(f"Hyper-params: {best_cfg}")
    print(f"Elapsed     : {elapsed/60:.1f} min")
    print("==========================")

    # Save full ranking (all configs) into a single params file
    ranking = []
    for score, cfg in results:
        # Strip out boilerplate / non-hp keys
        hp = {
            k: v for k, v in cfg.items()
            if k not in {"tree_method", "predictor", "device", "verbosity",
                         "objective", "eval_metric"}
        }
        ranking.append({
            "rmse": score,
            "params": hp,
        })

    out_params = save_root / f"best_{args.model}_{args.regressor}_params.json"
    with out_params.open("w") as f:
        json.dump(ranking, f, indent=2)
    logging.info("HP configs written to %s", out_params)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)