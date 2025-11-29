from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EARLY_STOP_ROUNDS = 20
VAL_FRACTION = 0.1

class TreeRegressor:
    """Huber-regression wrapper for XGBoost, LightGBM, and CatBoost back-ends."""

    def __init__(
        self,
        model_name: str | None,
        hyper: Optional[Dict[str, Any]],
    ) -> None:
        self._name = (model_name or "xgboost").lower()
        self._hyper = hyper or {}
        self._model: Any | None = None
        self._is_fitted: bool = False
        self._default_delta = self._infer_default_delta()

        if self._name == "xgboost":
            self._init_xgboost()
        elif self._name == "lightgbm":
            self._init_lightgbm()
        elif self._name == "catboost":
            self._init_catboost()
        elif self._name == "mlp":
            self._init_mlp()
        elif self._name == "linreg":
            self._init_linreg()
        else:
            logging.warning(
                "[Bliss] Unknown regressor '%s' – defaulting to XGBoost", self._name
            )
            self._name = "xgboost"
            self._init_xgboost()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "name": self._name,
            "hyper": self._hyper,
            "is_fitted": self._is_fitted,
        }
        if self._model is None:
            return state
        if self._name == "xgboost":
            booster = self._model.get_booster()
            state["model_raw"] = booster.save_raw()
        else:
            state["model_pickle"] = pickle.dumps(self._model)
        return state

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        name = state.get("name", self._name)
        hyper = state.get("hyper", self._hyper)
        # Backward-compatibility: if an old state passed huber_delta, inject into hyper if needed
        legacy_delta = state.get("huber_delta", None)
        if legacy_delta is not None:
            hyper = dict(hyper)
            if name == "xgboost" and "huber_slope" not in hyper:
                hyper["huber_slope"] = legacy_delta
            if name == "lightgbm" and "alpha" not in hyper:
                hyper["alpha"] = legacy_delta
            if name == "catboost" and "loss_function" not in hyper:
                hyper["loss_function"] = f"Huber:delta={legacy_delta}"
        self.__init__(name, hyper)

        if self._name == "xgboost":
            raw = state.get("model_raw")
            if raw is not None:
                self._model.load_model(bytearray(raw))
                self._is_fitted = True
        else:
            blob = state.get("model_pickle")
            if blob is not None:
                self._model = pickle.loads(blob)
                self._is_fitted = True
        self._is_fitted = bool(state.get("is_fitted", self._is_fitted))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_idx: Sequence[int] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if X.size == 0 or y.size == 0:
            return
        backend = self._name
        categorical_idx = tuple(sorted(categorical_idx or ()))

        X_np = np.asarray(X)
        y_np = np.asarray(y)
        w_np = np.asarray(sample_weight) if sample_weight is not None else None
        tr_split, va_split = self._train_val_split(X_np, y_np, w_np)

        if backend == "xgboost":
            self._fit_xgboost(tr_split, va_split, categorical_idx)
        elif backend == "lightgbm":
            self._fit_lightgbm(tr_split, va_split, categorical_idx)
        elif backend == "catboost":
            self._fit_catboost(tr_split, va_split, categorical_idx)
        elif backend == "mlp":
            self._fit_mlp(tr_split, va_split)
        elif backend == "linreg":
            self._fit_linreg(tr_split)
        else:
            return

        self._is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        categorical_idx: Sequence[int] | None = None,
    ) -> np.ndarray:
        if X.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._model is None or not self._is_fitted:
            return np.zeros(X.shape[0], dtype=np.float32)

        categorical_idx = tuple(sorted(categorical_idx or ()))
        backend = self._name

        if backend == "xgboost":
            X_df = self._as_dataframe(X, categorical_idx)
            preds = self._model.predict(X_df)
        elif backend == "lightgbm":
            X_df = self._as_dataframe(X, categorical_idx)
            preds = self._model.predict(X_df)
        elif backend == "catboost":
            X_df = self._as_dataframe(X, categorical_idx)
            preds = self._model.predict(X_df)
        elif backend in {"mlp", "linreg"}:
            preds = self._model.predict(X)
        else:
            preds = np.zeros(X.shape[0], dtype=np.float32)

        return np.asarray(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_required_packages(self, pkg: str) -> None:
        logging.debug("[Bliss] Initialising %s regressor", pkg)

    def _infer_default_delta(self, fallback: float = 2000.0) -> float:
        name = self._name
        h = self._hyper or {}
        try:
            if name == "xgboost":
                return float(h.get("huber_slope", fallback))
            if name == "lightgbm":
                return float(h.get("alpha", fallback))
            if name == "catboost":
                lf = h.get("loss_function")
                if isinstance(lf, str) and "delta=" in lf:
                    after = lf.split("delta=", 1)[1]
                    return float(after)
                return float(h.get("delta", fallback))
        except Exception:
            return float(fallback)
        return float(fallback)

    def _init_xgboost(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            logging.error("[Bliss] xgboost not installed – please `pip install xgboost`")
            raise exc

        params: Dict[str, Any] = {
            "tree_method": "hist",
            "objective": "reg:pseudohubererror",
            "huber_slope": self._default_delta,
            "enable_categorical": True,
            "device": "cpu",
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
            "eval_metric": "rmse",
        }
        params.update(self._hyper)
        # ensure Huber objective is set
        params["objective"] = "reg:pseudohubererror"
        params["huber_slope"] = params.get("huber_slope", self._default_delta)
        params.setdefault("enable_categorical", True)
        # Drop predictor to avoid unused-parameter warnings on some XGBoost builds
        params.pop("predictor", None)
        self._model = XGBRegressor(**params)

    def _init_lightgbm(self) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            logging.error(
                "[Bliss] lightgbm not installed – please `pip install lightgbm`"
            )
            raise exc

        params: Dict[str, Any] = {
            "objective": "huber",
            "alpha": self._default_delta,
            "verbosity": -1,
            "n_estimators": 200,
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
        }
        params.update(self._hyper)
        params["objective"] = "huber"
        params["alpha"] = params.get("alpha", self._default_delta)
        self._model = LGBMRegressor(**params)

    def _init_catboost(self) -> None:
        try:
            from catboost import CatBoostRegressor
        except ModuleNotFoundError as exc:
            logging.error(
                "[Bliss] catboost not installed – please `pip install catboost`"
            )
            raise exc

        base_loss = f"Huber:delta={self._default_delta}"
        params: Dict[str, Any] = {
            "loss_function": base_loss,
            "verbose": False,
            "iterations": 500,
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
        }
        params.update(self._hyper)
        params.setdefault("loss_function", base_loss)
        params.setdefault("verbose", False)
        self._model = CatBoostRegressor(**params)

    def _init_mlp(self) -> None:
        try:
            from sklearn.neural_network import MLPRegressor
        except ModuleNotFoundError as exc:
            logging.error(
                "[Bliss] scikit-learn not installed – please `pip install scikit-learn`"
            )
            raise exc

        params: Dict[str, Any] = {
            "early_stopping": True,
            "n_iter_no_change": 10,
            "validation_fraction": VAL_FRACTION,
        }
        params.update(self._hyper)
        params.setdefault("early_stopping", True)
        params.setdefault("n_iter_no_change", 10)
        params.setdefault("validation_fraction", VAL_FRACTION)
        self._model = MLPRegressor(**params)

    def _init_linreg(self) -> None:
        try:
            from sklearn.linear_model import LinearRegression
        except ModuleNotFoundError as exc:
            logging.error(
                "[Bliss] scikit-learn not installed – please `pip install scikit-learn`"
            )
            raise exc
        self._model = LinearRegression(**self._hyper)

    def _as_dataframe(
        self, X: np.ndarray, categorical_idx: Sequence[int]
    ):  # -> pandas.DataFrame
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "[Bliss] pandas is required for categorical regressors"
            ) from exc

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(np.asarray(X))

        for idx in categorical_idx:
            if idx >= df.shape[1]:
                continue
            df[idx] = pd.Categorical(df[idx].astype("int64"))
        return df

    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        val_fraction: float = VAL_FRACTION,
        seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]]:
        n = y.shape[0]
        if val_fraction <= 0 or n < 3:
            return (X, y, sample_weight), None

        val_size = max(1, int(n * val_fraction))
        if val_size >= n:
            return (X, y, sample_weight), None

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        va_idx = perm[:val_size]
        tr_idx = perm[val_size:]
        if tr_idx.size == 0:
            return (X, y, sample_weight), None

        def _split(arr: np.ndarray | None, idx: np.ndarray):
            if arr is None:
                return None
            return arr[idx]

        train = (X[tr_idx], y[tr_idx], _split(sample_weight, tr_idx))
        val = (X[va_idx], y[va_idx], _split(sample_weight, va_idx))
        return train, val

    def _fit_xgboost(
        self,
        train_split: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        val_split: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        categorical_idx: Sequence[int],
    ) -> None:
        X_tr, y_tr, w_tr = train_split
        X_df_tr = self._as_dataframe(X_tr, categorical_idx)
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if w_tr is not None:
            fit_kwargs["sample_weight"] = w_tr

        original_es = self._model.get_params().get("early_stopping_rounds", None)
        if val_split is not None:
            X_va, y_va, w_va = val_split
            X_df_va = self._as_dataframe(X_va, categorical_idx)
            fit_kwargs["eval_set"] = [(X_df_va, y_va)]
        else:
            if original_es is not None:
                self._model.set_params(early_stopping_rounds=None)

        self._model.fit(X_df_tr, y_tr, **fit_kwargs)

        if val_split is None and original_es is not None:
            self._model.set_params(early_stopping_rounds=original_es)

    def _fit_lightgbm(
        self,
        train_split: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        val_split: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        categorical_idx: Sequence[int],
    ) -> None:
        from lightgbm import early_stopping as lgb_early_stop

        X_tr, y_tr, w_tr = train_split
        X_df_tr = self._as_dataframe(X_tr, categorical_idx)
        fit_kwargs: Dict[str, Any] = {}
        if categorical_idx:
            fit_kwargs["categorical_feature"] = list(categorical_idx)
        if w_tr is not None:
            fit_kwargs["sample_weight"] = w_tr

        callbacks = []
        if val_split is not None:
            X_va, y_va, w_va = val_split
            X_df_va = self._as_dataframe(X_va, categorical_idx)
            fit_kwargs["eval_set"] = [(X_df_va, y_va)]
            fit_kwargs["eval_metric"] = "rmse"
            if w_va is not None:
                fit_kwargs["eval_sample_weight"] = [w_va]
            es_rounds = self._model.get_params().get("early_stopping_rounds", EARLY_STOP_ROUNDS)
            callbacks = [lgb_early_stop(stopping_rounds=es_rounds, verbose=False)]

        self._model.fit(X_df_tr, y_tr, callbacks=callbacks, **fit_kwargs)

    def _fit_catboost(
        self,
        train_split: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        val_split: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        categorical_idx: Sequence[int],
    ) -> None:
        X_tr, y_tr, w_tr = train_split
        X_df_tr = self._as_dataframe(X_tr, categorical_idx)
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if categorical_idx:
            fit_kwargs["cat_features"] = list(categorical_idx)
        if w_tr is not None:
            fit_kwargs["sample_weight"] = w_tr

        if val_split is not None:
            X_va, y_va, w_va = val_split
            X_df_va = self._as_dataframe(X_va, categorical_idx)
            fit_kwargs["eval_set"] = (X_df_va, y_va)
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = self._model.get_params().get(
                "early_stopping_rounds", EARLY_STOP_ROUNDS
            )
            if w_va is not None:
                fit_kwargs["eval_sample_weight"] = w_va

        self._model.fit(X_df_tr, y_tr, **fit_kwargs)

    def _fit_mlp(
        self,
        train_split: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        val_split: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> None:
        X_tr, y_tr, _ = train_split
        # MLPRegressor handles early stopping internally via validation_fraction
        self._model.fit(X_tr, y_tr)

    def _fit_linreg(
        self,
        train_split: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    ) -> None:
        X_tr, y_tr, _ = train_split
        self._model.fit(X_tr, y_tr)

    @staticmethod
    def _feature_types(
        n_features: int, categorical_idx: Sequence[int]
    ) -> List[str]:
        cats = set(int(i) for i in categorical_idx)
        return ["categorical" if i in cats else "float" for i in range(n_features)]
