#!/usr/bin/env python3
"""
GA hyper-parameter search for Bliss regressors using a rank-based Utility@K objective.

Usage example:
  python thirdparty/bliss/regressor_test/hyper_param_ga_search.py \
      --data_dir thirdparty/bliss/regressor_test/datasets/open_image_hp_collection \
      --model xgboost --regressor g --top_k 100 \
      --round_stride 3 --max_round 400
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer, Categorical
import sklearn_genetic.genetic_search as _ga_mod

# Tell sklearn-genetic that our stub is a regressor
_ga_mod.is_regressor = lambda est: isinstance(est, RegressorMixin)
_ga_mod.is_classifier = lambda est: False
_ga_mod.is_outlier_detector = lambda est: False

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore

EARLY_STOP_ROUNDS = 20
MIN_TRAIN_SAMPLES = 10
VAL_FRAC = 0.1


@dataclass
class RoundDataset:
    round_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    sample_rounds: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class BlissHPConfig(BaseEstimator):
    """Stub estimator for GASearchCV; training happens inside the custom scorer."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.__dict__.copy()

    def set_params(self, **params: Any) -> "BlissHPConfig":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> "BlissHPConfig":  # noqa: ARG002
        # Real fitting is performed inside the GA scorer.
        return self


class BlissHPRegressor(BlissHPConfig, RegressorMixin):
    """Small wrapper so sklearn_genetic sees this as a valid regressor."""
    pass


def compute_recency_weights(sample_rounds: np.ndarray, current_round: int, gamma: float) -> np.ndarray:
    if gamma is None or gamma <= 0:
        return np.ones_like(sample_rounds, dtype=np.float32)

    delta = np.maximum(0.0, current_round - sample_rounds.astype(np.float32))
    w = np.exp(-gamma * delta)
    mean_w = w.mean()
    if mean_w > 0:
        w /= mean_w
    else:
        w = np.ones_like(sample_rounds, dtype=np.float32)
    return w.astype(np.float32)


def find_latest_csv(base_dir: Path, subdir: str, regressor: str) -> Path:
    dir_path = base_dir / subdir
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing directory: {dir_path}")
    candidates = list(dir_path.glob(f"{regressor}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {regressor}_*.csv found under {dir_path}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logging.info("Using %s", latest)
    return latest


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    round_id: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    val_size = max(1, int(np.ceil(VAL_FRAC * n)))
    tr_size = n - val_size
    if tr_size < 1:
        raise ValueError("Not enough samples after validation split")

    rng = np.random.default_rng(random_state + int(round_id))
    idx = rng.permutation(n)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    return (
        X[tr_idx],
        X[val_idx],
        y[tr_idx],
        y[val_idx],
        weights[tr_idx],
        weights[val_idx],
    )


def instantiate_model(model_name: str, params: Dict[str, Any], random_state: int):
    model_name = model_name.lower()
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as e:  # pragma: no cover - import guard
            raise ImportError("Please install xgboost to use --model xgboost") from e

        cfg = dict(
            n_estimators=300,
            objective="reg:pseudohubererror",
            eval_metric="rmse",
            tree_method="hist",
            verbosity=0,
            random_state=random_state,
        )
        cfg.update(params)
        return XGBRegressor(**cfg)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor, early_stopping as lgb_early_stop
        except ModuleNotFoundError as e:  # pragma: no cover - import guard
            raise ImportError("Please install lightgbm to use --model lightgbm") from e
        cfg = dict(
            boosting_type="gbdt",
            objective="huber",
            n_estimators=300,
            verbosity=-1,
            random_state=random_state,
        )
        cfg.update(params)
        model = LGBMRegressor(**cfg)
        callbacks = [lgb_early_stop(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)]
        return model, callbacks

    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ModuleNotFoundError as e:  # pragma: no cover - import guard
            raise ImportError("Please install catboost to use --model catboost") from e
        huber_delta = params.pop("huber_delta", None)
        if huber_delta is None:
            raise ValueError("huber_delta is required for catboost")
        loss_function = f"Huber:delta={huber_delta}"
        cfg = dict(
            iterations=300,
            loss_function=loss_function,
            verbose=False,
            random_seed=random_state,
        )
        cfg.update(params)
        return CatBoostRegressor(**cfg)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_config_for_all_rounds(
    model_name: str,
    params: Dict[str, Any],
    rounds_data: List[RoundDataset],
    top_k: int,
    random_state: int,
    use_tqdm: bool,
) -> Tuple[float, float]:
    """
    Return (mean_utility_ratio, mean_rmse) over all usable rounds.

    mean_utility_ratio = average_r [ Utility@K_r(config) / Utility@K_r(oracle) ]
    mean_rmse          = average_r [ RMSE(y_pred, y_true) ] over same rounds.
    """
    recency_gamma = float(params.get("recency_gamma", 0.0) or 0.0)
    model_params = {k: v for k, v in params.items() if k != "recency_gamma"}

    ratio_scores: List[float] = []
    rmse_scores: List[float] = []

    iterator = rounds_data
    if use_tqdm and tqdm is not None:
        iterator = tqdm(rounds_data, desc="rounds", leave=False)

    for rd in iterator:
        if rd.X_train.shape[0] < MIN_TRAIN_SAMPLES or rd.X_test.shape[0] < 2:
            continue

        weights = compute_recency_weights(rd.sample_rounds, rd.round_id, recency_gamma)
        try:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = split_train_val(
                rd.X_train, rd.y_train, weights, rd.round_id, random_state
            )
        except ValueError:
            continue

        # Fit model
        if model_name == "xgboost":
            model = instantiate_model(model_name, model_params.copy(), random_state)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            X_test_input = rd.X_test  # numpy array

        elif model_name == "lightgbm":
            model_obj, callbacks = instantiate_model(model_name, model_params.copy(), random_state)

            # Use DataFrames consistently so LightGBM sees feature names both at fit and predict time
            n_features = X_tr.shape[1]
            cols = [f"f{i}" for i in range(n_features)]

            X_tr_df = pd.DataFrame(X_tr, columns=cols)
            X_val_df = pd.DataFrame(X_val, columns=cols)
            X_test_df = pd.DataFrame(rd.X_test, columns=cols)

            model_obj.fit(
                X_tr_df,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val_df, y_val)],
                eval_metric="rmse",
                callbacks=callbacks,
            )
            model = model_obj
            X_test_input = X_test_df  # DataFrame for predict

        else:  # catboost
            model = instantiate_model(model_name, model_params.copy(), random_state)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose=False,
            )
            X_test_input = rd.X_test  # numpy array

        # Predict using the same type used at fit time for each model
        y_pred = model.predict(X_test_input)
        y_true = rd.y_test

        # Utility@K / OracleUtility@K
        K_r = min(top_k, y_true.shape[0])
        if K_r < 1:
            continue

        idx_oracle = np.argsort(y_true)[::-1][:K_r]
        util_oracle = float(y_true[idx_oracle].sum())
        if util_oracle <= 0:
            continue

        idx_pred = np.argsort(y_pred)[::-1][:K_r]
        util_pred = float(y_true[idx_pred].sum())

        # Numerical safety: util_pred should theoretically be <= util_oracle,
        # but tiny FP differences can cause overshoot. Clip instead of crashing.
        if util_pred > util_oracle:
            overshoot = util_pred - util_oracle
            if overshoot > 1e-4:
                logging.warning(
                    "Large util_pred > util_oracle in round %d: "
                    "util_pred=%.10f, util_oracle=%.10f, overshoot=%.2e",
                    rd.round_id, util_pred, util_oracle, overshoot,
                )
            util_pred = util_oracle

        ratio_scores.append(util_pred / util_oracle)

        # RMSE on the full test set for that round
        rmse_r = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        rmse_scores.append(rmse_r)

    if not ratio_scores:
        # No usable rounds; degenerate metrics
        return 0.0, float("inf")

    mean_ratio = float(np.mean(ratio_scores))
    mean_rmse = float(np.mean(rmse_scores)) if rmse_scores else float("inf")
    return mean_ratio, mean_rmse


def build_param_grid(model: str) -> Dict[str, Any]:
    recency = {"recency_gamma": Continuous(1e-6, 0.01, distribution="log-uniform")}
    if model == "xgboost":
        grid = {
            **recency,
            "learning_rate": Continuous(0.01, 0.2, distribution="log-uniform"),
            "max_depth": Integer(3, 20),
            "subsample": Continuous(0.6, 1.0),
            "colsample_bytree": Continuous(0.6, 1.0),
            "min_child_weight": Integer(1, 10),
            "reg_alpha": Continuous(1e-6, 1.0, distribution="log-uniform"),
            "reg_lambda": Continuous(1e-6, 10.0, distribution="log-uniform"),
            "huber_slope": Integer(500, 3000),
        }
    elif model == "lightgbm":
        grid = {
            **recency,
            "num_leaves": Integer(15, 250),
            "learning_rate": Continuous(0.01, 0.2, distribution="log-uniform"),
            "min_child_samples": Integer(2, 50),
            "colsample_bytree": Continuous(0.6, 1.0),
            "subsample": Continuous(0.6, 1.0),
            "subsample_freq": Integer(1, 5),
            "reg_lambda": Continuous(1e-6, 5.0, distribution="log-uniform"),
            "alpha": Integer(500, 3000),
        }
    elif model == "catboost":
        grid = {
            **recency,
            "learning_rate": Continuous(0.01, 0.2, distribution="log-uniform"),
            "depth": Integer(3, 20),
            "l2_leaf_reg": Continuous(1.0, 30.0, distribution="log-uniform"),
            "bagging_temperature": Continuous(0.0, 2.0),
            "subsample": Continuous(0.7, 1.0),
            "rsm": Continuous(0.7, 1.0),
            "min_data_in_leaf": Integer(1, 30),
            "huber_delta": Integer(500, 3000),
        }
    else:
        raise ValueError(f"Unsupported model: {model}")
    return grid


def to_serializable(val: Any) -> Any:
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val


def summarize_top_configs(
    configs: List[Dict[str, Any]],
    tuned_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in tuned_keys:
        vals = [c["params"][key] for c in configs if key in c["params"]]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def prepare_round_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_rounds: np.ndarray,
) -> List[RoundDataset]:
    feat_cols = [c for c in train_df.columns if c.startswith("f")]
    X_train_all = train_df[feat_cols].to_numpy(np.float32)
    y_train_all = train_df["target_utility"].to_numpy(np.float32)
    train_round_all = train_df["round"].to_numpy(np.int32)
    sample_round_all = train_df["f17"].to_numpy(np.float32)

    X_test_all = test_df[feat_cols].to_numpy(np.float32)
    y_test_all = test_df["target_utility"].to_numpy(np.float32)
    test_round_all = test_df["round"].to_numpy(np.int32)

    rounds_data: List[RoundDataset] = []
    for r in selected_rounds:
        train_mask = train_round_all == r
        test_mask = test_round_all == r
        if not train_mask.any() or not test_mask.any():
            continue
        rounds_data.append(
            RoundDataset(
                round_id=int(r),
                X_train=X_train_all[train_mask],
                y_train=y_train_all[train_mask],
                sample_rounds=sample_round_all[train_mask],
                X_test=X_test_all[test_mask],
                y_test=y_test_all[test_mask],
            )
        )
    return rounds_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train/ and test/ CSVs")
    parser.add_argument("--model", required=True, choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--regressor", required=True, choices=["g", "h"])
    parser.add_argument("--top_k", type=int, required=True, help="K for Utility@K")
    parser.add_argument("--max_round", type=int, default=None, help="If set, only use rounds <= max_round")
    parser.add_argument("--round_stride", type=int, default=1, help="Use every k-th round (default=1)")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path("thirdparty/bliss/regressor_test/hp_configs")),
        help="Directory to store GA results.",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Optional job name; defaults to last component of data_dir.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for GA and model splits (default=42).",
    )
    args = parser.parse_args()

    if args.round_stride < 1:
        raise ValueError(f"round_stride must be >= 1, got {args.round_stride}")
    if args.top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {args.top_k}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_dir)
    train_path = find_latest_csv(data_root, "train", args.regressor)
    test_path = find_latest_csv(data_root, "test", args.regressor)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Feature alignment sanity check
    train_feats = {c for c in train_df.columns if c.startswith("f")}
    test_feats = {c for c in test_df.columns if c.startswith("f")}
    if train_feats != test_feats:
        raise ValueError("Train/test feature columns do not match")
    if "f17" not in train_feats:
        raise ValueError("Training CSV must contain f17 for recency weighting")

    unique_rounds = np.sort(test_df["round"].unique())
    if args.max_round is not None:
        unique_rounds = unique_rounds[unique_rounds <= args.max_round]
    selected_rounds = unique_rounds[:: args.round_stride]

    test_df = test_df[test_df["round"].isin(selected_rounds)]
    train_df = train_df[train_df["round"].isin(selected_rounds)]

    rounds_data = prepare_round_datasets(train_df, test_df, selected_rounds)
    if not rounds_data:
        raise ValueError("No overlapping rounds between train and test after filtering")
    num_rounds_used = len(rounds_data)

    param_grid = build_param_grid(args.model)
    tuned_keys = list(param_grid.keys())

    # Cache both (utility_ratio, rmse) per hyper-parameter config
    metric_cache: Dict[Tuple[Tuple[str, Any], ...], Tuple[float, float]] = {}

    def ga_scorer(estimator: BlissHPConfig, X_dummy: np.ndarray, y_dummy: np.ndarray) -> float:  # noqa: ARG001
        params = estimator.get_params(deep=False)
        key = tuple(sorted((k, to_serializable(v)) for k, v in params.items()))
        if key in metric_cache:
            ratio, rmse = metric_cache[key]
        else:
            ratio, rmse = evaluate_config_for_all_rounds(
                args.model,
                params,
                rounds_data,
                args.top_k,
                args.random_state,
                use_tqdm=(tqdm is not None),
            )
            metric_cache[key] = (ratio, rmse)
        return ratio  # GA optimises the utility ratio

    population_size = 2
    generations = 2
    n_jobs = 1

    ga = GASearchCV(
        estimator=BlissHPRegressor(),
        param_grid=param_grid,
        scoring=ga_scorer,
        cv=2,
        population_size=population_size,
        generations=generations,
        n_jobs=n_jobs,
        verbose=True,
    )

    # Dummy X/y just to satisfy sklearn's API; all scoring uses rounds_data
    X_dummy = np.zeros((4, 1), dtype=np.float32)
    y_dummy = np.zeros(4, dtype=np.float32)
    ga.fit(X_dummy, y_dummy)

    cv_results = ga.cv_results_
    scores = cv_results["mean_test_score"]
    params_list = cv_results["params"]

    # Sort configs by GA score (utility ratio)
    configs = sorted(
        [
            (float(s), p)
            for s, p in zip(scores, params_list, strict=False)
        ],
        key=lambda t: t[0],
        reverse=True,
    )

    num_configs = len(configs)
    top_n = min(30, num_configs)
    top_configs: List[Dict[str, Any]] = []

    for rank, (score, cfg) in enumerate(configs[:top_n], start=1):
        # Prepare key for metric cache
        key = tuple(sorted((k, to_serializable(v)) for k, v in cfg.items()))
        if key in metric_cache:
            ratio, rmse = metric_cache[key]
        else:
            ratio, rmse = evaluate_config_for_all_rounds(
                args.model,
                cfg,
                rounds_data,
                args.top_k,
                args.random_state,
                use_tqdm=False,
            )
            metric_cache[key] = (ratio, rmse)

        hp_only = {k: to_serializable(cfg[k]) for k in tuned_keys if k in cfg}
        top_configs.append(
            {
                "rank": rank,
                "score": float(ratio),  # same as utility_ratio
                "rmse": float(rmse),
                "params": hp_only,
            }
        )

    if not top_configs:
        raise RuntimeError("GA did not evaluate any configurations")

    best_score = top_configs[0]["score"]
    best_rmse = top_configs[0]["rmse"]
    best_params = top_configs[0]["params"]

    summary_stats = summarize_top_configs(top_configs, tuned_keys)

    job_name = args.job_name or data_root.name
    save_root = Path(args.save_dir) / job_name
    save_root.mkdir(parents=True, exist_ok=True)
    out_path = save_root / f"best_{args.model}_{args.regressor}_ga_params.json"

    payload = {
        "model": args.model,
        "regressor": args.regressor,
        "top_k": args.top_k,
        "round_stride": args.round_stride,
        "max_round": args.max_round,
        "random_state": args.random_state,
        "meta": {
            "num_rounds_used": num_rounds_used,
            "num_configs_evaluated": num_configs,
            "ga": {
                "population_size": population_size,
                "generations": generations,
                "n_jobs": n_jobs,
            },
        },
        "best": {
            "score": best_score,
            "rmse": best_rmse,
            "params": best_params,
        },
        "top_configs": top_configs,
        "summary_stats_top30": summary_stats,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logging.info("Best GA parameters written to %s", out_path)

    print("\n========== GA SEARCH RESULT ==========")
    print(f"Model       : {args.model}")
    print(f"Regressor   : {args.regressor}")
    print(f"top_k       : {args.top_k}")
    print(f"round_stride: {args.round_stride}")
    print(f"max_round   : {args.max_round}")
    print(f"Best score  : {best_score:.6f}")
    print(f"Best RMSE   : {best_rmse:.6f}")
    print(f"Configs eval: {num_configs}")
    print(f"Rounds used : {num_rounds_used}")
    print("======================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
