# thirdparty/bliss/regressor.py
from __future__ import annotations
import logging
import importlib
from typing import Any, Dict

import numpy as np


class Regressor:
    """
    Thin wrapper exposing the *same* `.fit` / `.predict` API for several
    back-ends.

    Parameters
    ----------
    model_name : str
        'linreg' | 'xgboost' | 'mlp'
    hyper      : dict | None
        Optional kwargs forwarded verbatim to the concrete model constructor.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model_name: str | None = "linreg",
        hyper: Dict[str, Any] | None = None,
    ) -> None:

        self._name   = (model_name or "linreg").lower()
        self._hyper  = hyper or {}
        self._model: Any | None = None

        if self._name == "xgboost":
            self._init_xgboost()

        elif self._name == "mlp":
            self._init_mlp()

        elif self._name == "linreg":
            self._init_linreg()

        else:
            logging.warning(
                "[Bliss] unknown regressor '%s' – defaulting to LinearRegression",
                self._name,
            )
            self._name = "linreg"
            self._init_linreg()

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """
        Train (or re-train) the model.  `sample_weight` is forwarded if the
        back-end supports it.
        """
        if X.size == 0:
            return

        try:
            self._model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            # back-end does not accept sample_weight
            self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros(0, dtype=np.float32)
        else:
            try:
                preds = self._model.predict(X)
                logging.info("[Regressor] Successfully predicted")
            except (AttributeError):
                preds = np.zeros(X.shape[0], dtype=np.float32)
                logging.info("[Regressor] Did not predict - not fitted error.")
        return preds.astype(np.float32)

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    # ---- 1. scikit-learn LinearRegression ----------------------------- #
    def _init_linreg(self) -> None:
        try:
            from sklearn.linear_model import LinearRegression
        except ModuleNotFoundError as e:
            logging.error(
                "[Bliss] scikit-learn not installed – please `pip install scikit-learn`"
            )
            raise e

        self._model = LinearRegression(**self._hyper)
        logging.info("[Bliss] LinearRegression initialised")

    # ---- 2. XGBoost --------------------------------------------------- #
    def _init_xgboost(self) -> None:
        try:
            XGBRegressor = importlib.import_module("xgboost").XGBRegressor
        except ModuleNotFoundError as e:
            logging.error("[Bliss] xgboost not installed – please `pip install xgboost`")
            raise e

        self._model = XGBRegressor(**self._hyper)
        logging.info("[Bliss] XGBoostRegressor initialised")

    # ---- 3. sklearn MLPRegressor ------------------------------------- #
    def _init_mlp(self) -> None:
        try:
            from sklearn.neural_network import MLPRegressor
        except ModuleNotFoundError as e:
            logging.error(
                "[Bliss] scikit-learn not installed – please `pip install scikit-learn`"
            )
            raise e

        self._model = MLPRegressor(**self._hyper)
        logging.info("[Bliss] MLPRegressor initialised")