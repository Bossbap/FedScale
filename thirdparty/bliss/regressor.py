# thirdparty/bliss/regressor.py
# ---------------------------------------------------------------
"""Small wrapper that lets Bliss plug-and-play different regressors.

Supported names
---------------
linear    – least–squares closed form (no external deps)
xgboost   – XGBRegressor from the xgboost package
mlp       – tiny torch MLP (hidden sizes configurable)

Example YAML
------------
predict_model: xgboost
predict_hyperparameters:
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 4

refresh_model: mlp
refresh_hyperparameters:
    hidden_sizes: [128, 64]
    epochs: 25
    lr: 0.001
    batch_size: 256
"""

from __future__ import annotations
import logging
import importlib
from typing import Any, Dict, List

import numpy as np


# ------------------------------------------------------------------ #
# 1) A super-light linear regressor (NumPy only)                     #
# ------------------------------------------------------------------ #
class _Linear:
    """Closed-form `y ≈ Xw + b` solved via `lstsq`."""
    def __init__(self, **_: Any) -> None:
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            self.w = None
            self.b = None
            return
        X_ = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        w_all, *_ = np.linalg.lstsq(X_, y, rcond=None)
        self.w = w_all[:-1]
        self.b = float(w_all[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        return X @ self.w + self.b


# ------------------------------------------------------------------ #
# 2) Tiny Torch MLP                                                  #
# ------------------------------------------------------------------ #
class _TinyMLP:
    """
    2-to-4 layer ReLU MLP with a linear output head.

    Hyper-parameters (with defaults)
    --------------------------------
    hidden_sizes : List[int]   e.g. [128, 64]
    epochs       : int         20
    lr           : float       1e-3
    batch_size   : int         256
    """

    def __init__(self,
                 hidden_sizes: List[int] | None = None,
                 epochs: int = 20,
                 lr: float = 1e-3,
                 batch_size: int = 256,
                 **_: Any) -> None:
        import torch
        from torch import nn

        hidden_sizes = hidden_sizes or [128, 64]

        layers: List[nn.Module] = []
        in_dim = None  # set at fit-time when we know X.shape[1]

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._bs = int(batch_size)
        self._layers_cfg = hidden_sizes
        self._model: nn.Module | None = None

    # -- helpers -----------------------------------------------------
    def _build(self, in_dim: int):
        import torch
        from torch import nn

        layers: List[nn.Module] = []
        prev = in_dim
        for h in self._layers_cfg:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))  # output scalar
        self._model = nn.Sequential(*layers).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader

        if X.size == 0:
            self._model = None
            return

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

        if self._model is None:
            self._build(X.shape[1])

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self._bs, shuffle=True)

        opt = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        loss_f = nn.MSELoss()
        self._model.train()

        for _ in range(self._epochs):
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                pred = self._model(xb)
                loss = loss_f(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._model is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self._device)
            out = self._model(xb).cpu().numpy().flatten()
        return out


# ------------------------------------------------------------------ #
# 3) Glue wrapper exposed to Bliss                                   #
# ------------------------------------------------------------------ #
class Regressor:
    """
    Unifies the interface of several regressors.

    Parameters
    ----------
    model_name : str
        'linear' | 'xgboost' | 'mlp'
    hyper      : dict | None
        Arbitrary kwargs forwarded to the concrete model.
    """

    def __init__(self,
                 model_name: str,
                 hyper: Dict[str, Any] | None = None) -> None:

        self._name = (model_name or "linear").lower()
        hyper = hyper or {}
        self._model: Any

        if self._name in {"linear", "linreg"}:
            self._model = _Linear(**hyper)

        elif self._name in {"xgb", "xgboost"}:
            try:
                XGBRegressor = importlib.import_module(
                    "xgboost").XGBRegressor
                # sensible defaults if none provided
                defaults = dict(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    n_jobs=1,
                )
                defaults.update(hyper)
                self._model = XGBRegressor(**defaults)
            except ModuleNotFoundError:
                logging.warning(
                    "[Bliss] xgboost not installed – falling back to linear")
                self._model = _Linear()

        elif self._name in {"mlp", "torch"}:
            self._model = _TinyMLP(**hyper)

        else:
            logging.warning(
                "[Bliss] unknown regressor '%s' – using linear", self._name)
            self._model = _Linear(**hyper)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train / re-train the underlying model."""
        if X.size == 0:
            return
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X (shape = (N, d))."""
        if X.size == 0:
            return np.zeros(0, dtype=np.float32)
        preds = self._model.predict(X).astype(np.float32)
        # Uncomment if you prefer hard non-negativity:
        # preds = np.maximum(0.0, preds)
        return preds
