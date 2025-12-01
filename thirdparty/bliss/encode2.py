from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


TRACE_KEYS: Tuple[str, ...] = ("rates", "availabilities", "batteryLevels")
TRACE_FEATURE_SUFFIXES: Tuple[str, ...] = ("mean", "std", "slope", "last")
_TRACE_TIME_AXIS = np.arange(5, dtype=np.float32)

NUM_DYNAMIC_FEATURES = len(TRACE_KEYS) * len(TRACE_FEATURE_SUFFIXES)  # 12
STATIC_NUMERIC_FEATURES = 3  # cpu_flops, gpu_flops, peak_throughput
NUM_CATEGORICAL_FEATURES = 2  # has_gpu, cluster_rank
ROUND_FEATURES = 1

CAT_FEATURE_IDX: Tuple[int, int] = (
    NUM_DYNAMIC_FEATURES + STATIC_NUMERIC_FEATURES,
    NUM_DYNAMIC_FEATURES + STATIC_NUMERIC_FEATURES + 1,
)


@dataclass
class EncodedBatch:
    """Container holding encoded features and book-keeping metadata."""

    matrix: np.ndarray
    ids: List[int]
    categorical_idx: Tuple[int, ...]


def _ensure_length(arr: Sequence[float] | np.ndarray, length: int = 5) -> np.ndarray:
    """Return a float32 copy padded/truncated to `length`."""
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return np.zeros(length, dtype=np.float32)
    if vec.size == length:
        return vec
    if vec.size > length:
        return vec[-length:]
    pad = np.repeat(vec[-1], length - vec.size)
    return np.concatenate([vec, pad]).astype(np.float32)


def _least_squares_slope(values: np.ndarray) -> float:
    """Slope of the best LS line through the values at t=0..len(values)-1."""
    if values.size == 0:
        return 0.0
    x = _TRACE_TIME_AXIS[: values.size]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(values))
    num = float(np.sum((x - x_mean) * (values - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    return num / den if den > 0 else 0.0


def _summarise_trace(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vec = _ensure_length(values)
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    slope = _least_squares_slope(vec)
    last = float(vec[-1]) if vec.size else 0.0
    return np.asarray([mean, std, slope, last], dtype=np.float32)


def _dynamic_features(dynamic_metadata: Dict[str, Any]) -> np.ndarray:
    stats: List[np.ndarray] = []
    for key in TRACE_KEYS:
        stats.append(_summarise_trace(dynamic_metadata.get(key, [])))
    return np.concatenate(stats).astype(np.float32)


def _static_numeric_features(static_metadata: Dict[str, Any]) -> np.ndarray:
    cpu = float(static_metadata.get("cpu_flops", 0.0) or 0.0)
    gpu = float(static_metadata.get("gpu_flops", 0.0) or 0.0)
    thr = float(static_metadata.get("peak_throughput", 0.0) or 0.0)
    return np.asarray([cpu, gpu, thr], dtype=np.float32)


def _categorical_features(static_metadata: Dict[str, Any]) -> np.ndarray:
    gpu_flops = float(static_metadata.get("gpu_flops", 0.0) or 0.0)
    has_gpu = 1 if gpu_flops > 0 else 0
    cluster_rank = static_metadata.get("cluster_rank", 0)
    try:
        cluster_rank_val = int(cluster_rank)
    except (TypeError, ValueError):
        cluster_rank_val = 0
    return np.asarray([has_gpu, cluster_rank_val], dtype=np.float32)


def encode_g(records: List[Dict[str, Any]]) -> EncodedBatch:
    """Encode new/unseen-client metadata into model-ready features."""
    if not records:
        return EncodedBatch(
            matrix=np.empty((0, 0), dtype=np.float32),
            ids=[],
            categorical_idx=CAT_FEATURE_IDX,
        )

    rows: List[np.ndarray] = []
    ids: List[int] = []
    for rec in records:
        dynamic = _dynamic_features(rec.get("dynamic_metadata", {}))
        static = _static_numeric_features(rec.get("static_metadata", {}))
        categorical = _categorical_features(rec.get("static_metadata", {}))
        round_idx = float(rec.get("round_index", rec.get("round", 0.0)) or 0.0)
        combined = np.concatenate([dynamic, static, categorical, np.asarray([round_idx], dtype=np.float32)]).astype(np.float32)
        rows.append(combined)
        ids.append(int(rec["client_id"]))

    matrix = np.stack(rows).astype(np.float32)
    return EncodedBatch(matrix=matrix, ids=ids, categorical_idx=CAT_FEATURE_IDX)


def encode_h(records: List[Dict[str, Any]]) -> EncodedBatch:
    """Encode returning clients (g-features + structured history statistics)."""
    base = encode_g(records)
    if base.matrix.size == 0:
        return base

    history_rows: List[List[float]] = []
    for rec in records:
        history = rec.get("history", {})
        n_participations = float(history.get("n_participations", 0.0) or 0.0)
        ema_u = float(history.get("ema_utility", 0.0) or 0.0)
        std_u = float(history.get("std_utility", 0.0) or 0.0)
        time_since_last = float(history.get("time_since_last", 0.0) or 0.0)
        success_rate = float(history.get("success_rate", 0.0) or 0.0)
        ema_norm_u = float(history.get("ema_norm_utility", 0.0) or 0.0)
        std_norm_u = float(history.get("std_norm_utility", 0.0) or 0.0)
        last_raw_u = float(history.get("last_raw_utility", 0.0) or 0.0)
        hist_rate_ema = float(history.get("historic_rate_mean_ema", 0.0) or 0.0)
        hist_avail_ema = float(history.get("historic_availability_mean_ema", 0.0) or 0.0)
        hist_batt_ema = float(history.get("historic_battery_mean_ema", 0.0) or 0.0)
        delta_rate = float(history.get("delta_rate_mean", 0.0) or 0.0)
        delta_avail = float(history.get("delta_availability_mean", 0.0) or 0.0)
        delta_batt = float(history.get("delta_battery_mean", 0.0) or 0.0)
        ratio_rate = float(history.get("ratio_rate_mean", 0.0) or 0.0)
        ratio_avail = float(history.get("ratio_availability_mean", 0.0) or 0.0)
        ratio_batt = float(history.get("ratio_battery_mean", 0.0) or 0.0)
        history_rows.append(
            [
                n_participations,
                ema_u,
                std_u,
                time_since_last,
                success_rate,
                ema_norm_u,
                std_norm_u,
                last_raw_u,
                hist_rate_ema,
                hist_avail_ema,
                hist_batt_ema,
                delta_rate,
                delta_avail,
                delta_batt,
                ratio_rate,
                ratio_avail,
                ratio_batt,
            ]
        )

    hist_matrix = np.asarray(history_rows, dtype=np.float32)
    matrix = np.concatenate([base.matrix, hist_matrix], axis=1)
    return EncodedBatch(matrix=matrix, ids=base.ids, categorical_idx=base.categorical_idx)
