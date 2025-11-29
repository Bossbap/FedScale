#!/usr/bin/env python3
"""
Ablation study driver for Bliss regressors (g/h).

example usage:
python thirdparty/bliss/regressor_test/ablation_study.py \
  thirdparty/bliss/regressor_test/datasets/openimage_bliss \
  h 400 2 5 --tree_model xgboost

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

EARLY_STOP_ROUNDS = 20
DEFAULT_SAVE_DIR = Path("thirdparty/bliss/regressor_test/ablation_result")


def resolve_xgb_device(preferred: str | None) -> tuple[str, str, str]:
    """Return a tuple (device, tree_method, predictor) compatible with XGBoost."""

    def _defaults(device_name: str) -> tuple[str, str, str]:
        is_cpu = device_name.lower() == "cpu"
        tree_method = "hist"
        predictor = "auto" if is_cpu else "gpu_predictor"
        return device_name, tree_method, predictor

    if preferred:
        return _defaults(preferred)

    return _defaults("cpu")


def load_latest_csv(data_dir: Path, regressor: str) -> pd.DataFrame:
    csvs = sorted(data_dir.glob(f"{regressor}_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No {regressor}_*.csv in {data_dir}")
    logging.info("Dataset: %s", csvs[-1].name)
    return pd.read_csv(csvs[-1])


def filter_rounds(df: pd.DataFrame, max_round: int | None, round_stride: int) -> pd.DataFrame:
    """Apply max_round then round stride filtering in-place style."""
    if max_round is not None:
        before = df.shape[0]
        df = df[df["round"] <= max_round]
        logging.info(
            "Applied max_round=%s: kept %d / %d rows",
            max_round,
            df.shape[0],
            before,
        )

    if round_stride > 1:
        unique_rounds = np.sort(df["round"].unique())
        selected_rounds = unique_rounds[::round_stride]
        before = df.shape[0]
        df = df[df["round"].isin(selected_rounds)]
        logging.info(
            "Applied round_stride=%d: kept %d / %d rows, %d rounds -> %d rounds",
            round_stride,
            df.shape[0],
            before,
            len(unique_rounds),
            len(selected_rounds),
        )
    return df


def get_default_params(model: str, tree_model: str, device: str) -> dict:
    """Return a single default hyper-parameter set per model."""
    tree_model = tree_model.lower()

    if tree_model == "xgboost" and model == "g":
        device_name, tree_method, predictor = resolve_xgb_device(device)
        return dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            subsample=0.7,
            colsample_bytree=1.0,
            min_child_weight=1,
            reg_alpha=0.0,
            reg_lambda=0.0,
            huber_slope=1000,
            tree_method=tree_method,
            predictor=predictor,
            device=device_name,
            objective="reg:pseudohubererror",
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbosity=0,
        )
    
    if tree_model == "xgboost" and model == "h":
        device_name, tree_method, predictor = resolve_xgb_device(device)
        return dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_alpha=0.0,
            reg_lambda=0.0,
            huber_slope=3000,
            tree_method=tree_method,
            predictor=predictor,
            device=device_name,
            objective="reg:pseudohubererror",
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbosity=0,
        )

    if tree_model == "lightgbm" and model == "g":
        return dict(
            boosting_type="gbdt",
            objective="huber",
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=8,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            alpha=2000,
            verbosity=-1,
        )
    
    if tree_model == "lightgbm" and model == "h":
        return dict(
            boosting_type="gbdt",
            objective="huber",
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=8,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            alpha=2000,
            verbosity=-1,
        )

    if tree_model == "catboost" and model == "g":
        return dict(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=10,
            bagging_temperature=0.5,
            loss_function="Huber:delta=2000",
            subsample=0.9,
            rsm=0.9,
        )
    
    if tree_model == "catboost" and model == "h":
        return dict(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=10,
            bagging_temperature=0.5,
            loss_function="Huber:delta=2000",
            subsample=0.9,
            rsm=0.9,
        )

    return dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=1e-3,
        batch_size=128,
        alpha=1e-4,
        solver="adam",
    )


def mean_round_rmse(y_true: np.ndarray, y_pred: np.ndarray, rounds: np.ndarray, cv_splits: int) -> float:
    rmses: list[float] = []
    for r in np.unique(rounds):
        mask = rounds == r
        if mask.sum() < cv_splits:
            continue
        mse_r = mean_squared_error(y_true[mask], y_pred[mask])
        rmses.append(float(np.sqrt(mse_r)))
    if not rmses:
        return float("inf")
    return float(np.mean(rmses))


def evaluate_subset(
    tree_model: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    rounds: np.ndarray,
    cv_splits: int,
    rng_seed: int,
    feature_names: list[str] | None = None,
) -> float:
    """Global KFold CV with out-of-fold predictions and per-round RMSE."""
    model_name = tree_model.lower()
    is_xgb = model_name == "xgboost"
    is_lgbm = model_name == "lightgbm"
    is_cat = model_name == "catboost"

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
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=rng_seed)

    for tr, va in kf.split(X):
        if is_xgb:
            model = XGBRegressor(**params)
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                verbose=False,
            )
            X_va = X[va]

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
                callbacks=[
                    lgb_early_stop(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)
                ],
            )

        elif is_cat:
            model = CatBoostRegressor(
                **params,
                random_seed=rng_seed,
                verbose=False,
            )
            model.fit(
                X[tr],
                y[tr],
                eval_set=(X[va], y[va]),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose=False,
            )
            X_va = X[va]

        else:
            es_kwargs = dict(
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.1,
            )
            # Avoid warnings when early stopping kicks in.
            model = MLPRegressor(
                **params,
                random_state=rng_seed,
                **es_kwargs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(X[tr], y[tr])
            X_va = X[va]

        y_pred[va] = model.predict(X_va).astype(np.float32)

    return mean_round_rmse(y, y_pred, rounds, cv_splits)


def build_ablation_configs(model: str) -> Dict[str, List[int]]:
    """Explicit mapping from ablation name to indices dropped."""
    configs: Dict[str, List[int]] = {
        "full": [],
        "full_minus_cluster_rank": [16],
        "full_minus_has_gpu": [15],
        "full_minus_round_index": [17],
        "full_minus_rate_summaries": [0, 1, 2, 3],
        "full_minus_availability_summaries": [4, 5, 6, 7],
        "full_minus_battery_summaries": [8, 9, 10, 11],
        "full_minus_all_dynamic_summaries": list(range(0, 12)),
        "full_minus_static_system": [12, 13, 14],
    }

    if model == "h":
        configs.update({
            "full_minus_round_index_time_since_last": [17, 21],
            "full_minus_history_block": list(range(18, 26)),
            "history_only": list(range(0, 18)),
            "full_minus_last_raw_u": [25],
            "full_minus_std_u_std_norm_u": [20, 24],
            "full_minus_ema_norm_u_std_norm_u": [23, 24],
            "full_minus_ema_u_std_u": [19, 20],
            "full_minus_ema_u_std_u_last_raw_u": [19, 20, 25],
            "full_minus_ema_norm_u_std_norm_u_last_raw_u": [23, 24, 25],
            "full_minus_ema_norm_u_std_norm_u_ema_u": [23, 24, 19],
        })

    # Dynamic summaries shape ablations per group
    groups = {
        "rate": [0, 1, 2, 3],
        "availability": [4, 5, 6, 7],
        "battery": [8, 9, 10, 11],
    }
    for name, idxs in groups.items():
        mean_i, std_i, slope_i, last_i = idxs
        configs[f"full_minus_{name}_mean"] = [mean_i]
        configs[f"full_minus_{name}_std"] = [std_i]
        configs[f"full_minus_{name}_slope"] = [slope_i]
        configs[f"full_minus_{name}_last"] = [last_i]
        configs[f"full_minus_{name}_mean_last"] = [mean_i, last_i]

    means = [idxs[0] for idxs in groups.values()]
    stds = [idxs[1] for idxs in groups.values()]
    slopes = [idxs[2] for idxs in groups.values()]
    lasts = [idxs[3] for idxs in groups.values()]
    configs["full_minus_all_means"] = means
    configs["full_minus_all_stds"] = stds
    configs["full_minus_all_slopes"] = slopes
    configs["full_minus_all_lasts"] = lasts
    configs["full_minus_all_mean_last"] = means + lasts

    return configs


def prepare_ablation_list(
    configs: Dict[str, List[int]],
    n_features: int,
    feat_cols: list[str],
) -> list[tuple[str, list[int], list[str]]]:
    """Filter out configs that do not apply to the current feature set."""
    ablations: list[tuple[str, list[int], list[str]]] = []
    for name, drop_idxs in configs.items():
        invalid = [i for i in drop_idxs if i < 0 or i >= n_features]
        if invalid:
            logging.warning("Skipping ablation %s; invalid indices: %s", name, invalid)
            continue
        drop_names = [feat_cols[i] if i < len(feat_cols) else f"f{i}" for i in sorted(drop_idxs)]
        ablations.append((name, drop_idxs, drop_names))
    return ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablations for Bliss regressors.")
    parser.add_argument("data_dir", help="Path to regressor_test/datasets/<job_name>")
    parser.add_argument("model", choices=["g", "h"], help="Regressor type to evaluate")
    parser.add_argument("max_round", type=int, help="Keep only rows with round <= max_round")
    parser.add_argument("round_stride", type=int, help="Use every k-th round (>=1)")
    parser.add_argument("cv_splits", type=int, help="Number of CV folds (>=2)")
    parser.add_argument(
        "--tree_model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "catboost", "mlp"],
        help="Model family to use for the ablation.",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Optional job name override (defaults to last component of data_dir).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="Base directory for ablation outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string for XGBoost (cpu or cuda:0, ...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits.",
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help="Optional JSON string to override default model params.",
    )
    args = parser.parse_args()

    if args.round_stride < 1:
        raise ValueError(f"round_stride must be >= 1, got {args.round_stride}")
    if args.cv_splits < 2:
        raise ValueError(f"cv_splits must be >= 2, got {args.cv_splits}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = load_latest_csv(Path(args.data_dir), args.model)
    df = filter_rounds(df, args.max_round, args.round_stride)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_full = df[feat_cols].to_numpy(np.float32)
    y = df["target_utility"].to_numpy(np.float32)
    rounds = df["round"].to_numpy(np.int32)
    n_features = X_full.shape[1]
    logging.info("Loaded %d samples with %d features", X_full.shape[0], n_features)

    model_params = get_default_params(args.model, args.tree_model, args.device)
    if args.params_json:
        overrides = json.loads(args.params_json)
        model_params.update(overrides)
        logging.info("Overriding params with %s", overrides)

    configs = build_ablation_configs(args.model)
    ablations = prepare_ablation_list(configs, n_features, feat_cols)

    if not ablations:
        raise RuntimeError("No valid ablation configs for this dataset.")

    job_name = args.job_name or Path(args.data_dir).name
    save_root = Path(args.save_dir) / job_name / args.model
    save_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for name, drop_idxs, drop_names in ablations:
        keep = [i for i in range(n_features) if i not in set(drop_idxs)]
        X_subset = X_full[:, keep]
        logging.info(
            "Evaluating %s: drop %s (features used: %d)",
            name,
            drop_names if drop_names else "none",
            X_subset.shape[1],
        )
        score = evaluate_subset(
            args.tree_model,
            model_params,
            X_subset,
            y,
            rounds,
            args.cv_splits,
            args.seed,
            [feat_cols[i] for i in keep],
        )
        results.append(
            dict(
                subset_name=name,
                dropped_features=drop_names,
                n_features_used=X_subset.shape[1],
                mean_round_rmse=score,
            )
        )

    results_df = pd.DataFrame(results).sort_values("mean_round_rmse").reset_index(drop=True)
    csv_path = save_root / f"{args.tree_model}_ablation_results.csv"
    json_path = save_root / f"{args.tree_model}_ablation_results.json"
    results_df.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results written to %s and %s", csv_path, json_path)

    baseline_row = results_df[results_df["subset_name"] == "full"]
    baseline_rmse = baseline_row["mean_round_rmse"].iloc[0] if not baseline_row.empty else None

    print("\n===== Ablation Summary =====")
    print(f"Job        : {job_name}")
    print(f"Regressor  : {args.model}")
    print(f"Tree model : {args.tree_model}")
    if baseline_rmse is not None:
        print(f"Baseline(full) RMSE: {baseline_rmse:.5f}")
    else:
        print("Baseline(full) RMSE: not available (baseline skipped)")
    top_k = results_df.head(5)
    print("\nTop ablations:")
    for _, row in top_k.iterrows():
        dropped = row["dropped_features"]
        dropped_str = ", ".join(dropped) if dropped else "none"
        print(f"- {row['subset_name']}: RMSE={row['mean_round_rmse']:.5f} (dropped: {dropped_str})")
    print("============================\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
