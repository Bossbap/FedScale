from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TreeRegressor:
    """Huber-regression wrapper for XGBoost, LightGBM, and CatBoost back-ends."""

    def __init__(
        self,
        model_name: str | None,
        hyper: Optional[Dict[str, Any]],
        huber_delta: float,
    ) -> None:
        self._name = (model_name or "xgboost").lower()
        self._hyper = hyper or {}
        self._huber_delta = float(huber_delta)
        self._model: Any | None = None
        self._is_fitted: bool = False

        if self._name == "xgboost":
            self._init_xgboost()
        elif self._name == "lightgbm":
            self._init_lightgbm()
        elif self._name == "catboost":
            self._init_catboost()
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
            "huber_delta": self._huber_delta,
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
        huber_delta = state.get("huber_delta", self._huber_delta)
        self.__init__(name, hyper, huber_delta)

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

        if backend == "xgboost":
            X_df = self._as_dataframe(X, categorical_idx)
            fit_kwargs: Dict[str, Any] = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self._model.fit(X_df, y, **fit_kwargs)

        elif backend == "lightgbm":
            X_df = self._as_dataframe(X, categorical_idx)
            fit_kwargs = {}
            if categorical_idx:
                fit_kwargs["categorical_feature"] = list(categorical_idx)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self._model.fit(X_df, y, **fit_kwargs)

        elif backend == "catboost":
            X_df = self._as_dataframe(X, categorical_idx)
            fit_kwargs = {}
            if categorical_idx:
                fit_kwargs["cat_features"] = list(categorical_idx)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self._model.fit(X_df, y, **fit_kwargs)
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
        else:
            preds = np.zeros(X.shape[0], dtype=np.float32)

        return np.asarray(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_required_packages(self, pkg: str) -> None:
        logging.debug("[Bliss] Initialising %s regressor", pkg)

    def _init_xgboost(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            logging.error("[Bliss] xgboost not installed – please `pip install xgboost`")
            raise exc

        params: Dict[str, Any] = {
            "tree_method": "hist",
            "objective": "reg:pseudohubererror",
            "huber_slope": self._huber_delta,
            "enable_categorical": True,
        }
        params.update(self._hyper)
        params["objective"] = "reg:pseudohubererror"
        params["huber_slope"] = self._huber_delta
        params["enable_categorical"] = True
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
            "alpha": self._huber_delta,
        }
        params.update(self._hyper)
        params["objective"] = "huber"
        params["alpha"] = self._huber_delta
        self._model = LGBMRegressor(**params)

    def _init_catboost(self) -> None:
        try:
            from catboost import CatBoostRegressor
        except ModuleNotFoundError as exc:
            logging.error(
                "[Bliss] catboost not installed – please `pip install catboost`"
            )
            raise exc

        base_loss = f"Huber:delta={self._huber_delta}"
        params: Dict[str, Any] = {
            "loss_function": base_loss,
            "verbose": False,
        }
        params.update(self._hyper)
        params["loss_function"] = base_loss
        params.setdefault("verbose", False)
        self._model = CatBoostRegressor(**params)

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

    @staticmethod
    def _feature_types(
        n_features: int, categorical_idx: Sequence[int]
    ) -> List[str]:
        cats = set(int(i) for i in categorical_idx)
        return ["categorical" if i in cats else "float" for i in range(n_features)]
