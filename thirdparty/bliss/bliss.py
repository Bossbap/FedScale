"""Bliss – Adaptive client‑selection strategy for FedScale.

This implementation follows the high‑level pipeline described in the PDF draft
(“Adaptive Client Selection Strategy in Cross‑Device Federated Learning”).  The
public API remains **identical** to Oort’s so that FedScale can switch between
`sample_mode: oort` and `sample_mode: bliss` without touching core code.

The predictive models `gθ` (utility drift) and `hϕ` (utility estimation for
unseen clients) are **stubbed with very lightweight linear regressors trained
via `numpy.linalg.lstsq`** so that they are fast, dependency‑free and keep the
shape of the algorithm.  Drop‑in replacement with more sophisticated models is
straight‑forward – just plug them behind the same method signatures.
"""

import logging
import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[2]   # adjust depth if needed
sys.path.append(str(repo_root))

from thirdparty.bliss.encode import encode_static_metadata

# -----------------------------------------------------------------------------
# Public factory helpers – FedScale expects these names
# -----------------------------------------------------------------------------

def create_training_selector(args):
    """Factory used by `ClientManager` during training‑time sampling."""
    return _training_selector(args)


def create_testing_selector(
    data_distribution: Optional[Dict[Any, Any]] = None,
    client_info: Optional[Dict[int, Sequence[float]]] = None,
    model_size: Optional[int] = None,
):
    """Factory for the testing selector (currently unused for Bliss)."""
    return _testing_selector(data_distribution, client_info, model_size)


# -----------------------------------------------------------------------------
# Training selector – core of Bliss
# -----------------------------------------------------------------------------

class _training_selector:
    """Bliss training‑phase selector (implements Algorithm 1 from the PDF)."""

    def __init__(self, args, sample_seed: int = 233):
        self.number_clients_to_predict_utility = args.number_clients_to_predict_utility
        self.number_clients_to_refresh_utility = args.number_clients_to_refresh_utility
        self.amount_clients_refresh_train_set = args.amount_clients_refresh_train_set
        self.amount_clients_predict_train_set = args.amount_clients_predict_train_set
        self.ema_alpha = args.ema_alpha
        self.rng = random.Random(sample_seed)
        self.round = 0

        # Per‑client metadata
        self.clients: Dict[int, Dict[str, Any]] = {}

        self.clients_to_predict = []
        self.clients_to_refresh = []

        # Buffers for (Δm, Δu) pairs used to train gθ
        self._drift_X: deque[np.ndarray] = deque(maxlen=self.amount_clients_refresh_train_set)
        self._drift_y: deque[float] = deque(maxlen=self.amount_clients_predict_train_set)
        self._g_w: Optional[np.ndarray] = None  # weights for gθ
        self._g_b: Optional[float] = None

        # Linear regressor weights (gθ and hϕ)
        self._g_w: np.ndarray | None = None
        self._g_b: float | None = None
        self._h_w: np.ndarray | None = None
        self._h_b: float | None = None

        logging.info("[Bliss] training selector ready (seed=%d)", sample_seed)

    # ------------------------------------------------------------------
    # Interface called by ClientManager / Aggregator
    # ------------------------------------------------------------------

    def register_client(self, client_id: int, feedbacks: Dict[str, Any]):
        """Add a new client to the system (initially considered *unseen*)."""
        if client_id in self.clients:
            logging.debug("[Bliss] Client %s already in seen set – skipping re‑register", client_id)
            return
        self.clients[client_id] = {
            'utility': 0.0,
            'last_utility': 0.0,
            'success': False,
            'last_success': False,
            'static_metadata':feedbacks.get('metadata'),
            'dynamic_metadata': {
                'rates': np.zeros(5),
                'availabilities': np.zeros(5),
                'batteryLevels': np.zeros(5),
            },
            'last_dynamic_metadata': {
                'availabilities': np.zeros(5),
                'rates': np.zeros(5),
                'batteryLevels': np.zeros(5),
            },
            'round': -1,
            'last_round': -1,
            'seen': 0
        }

    def update_client_metadata_pre_training(self, feedbacks: Dict[str, Any]):
        client_id = feedbacks['client_id']

        self.clients[client_id]['last_dynamic_metadata'] = self.clients[client_id]['dynamic_metadata']
        self.clients[client_id]['dynamic_metadata'] = feedbacks['dynamic_metadata']


    # Called once per round for *participating* clients
    def update_client_metadata_post_training(self, client_id: int, feedbacks: Dict[str, Any]):
        client = self.clients[client_id]
        
        util = feedbacks['reward']
        success = feedbacks['success']

        client['last_utility'] = client['utility']
        client['last_success'] = client['success']

        # EMA update
        client['utility'] = self.ema_alpha * util + (1 - self.ema_alpha) * client['last_utility']

        # No EMA
        # client['utility'] = util
        
        client['success'] = success
        client['last_round'] = self.round

    # ------------------------------------------------------------------
    # Weighted sampling helper
    # ------------------------------------------------------------------

    def _weighted_sample(self, pool_ids: List[int], k: int) -> List[int]:
        """Sample *k* distinct ids where P(id) ∝ utility(id)."""
        if len(pool_ids) <= k:
            return pool_ids.copy()

        util = np.array([self.clients[cid]["utility"] for cid in pool_ids], dtype=float)
        # Ensure strictly positive weights
        util = np.clip(util, 1e-6, None)
        probs = util / util.sum()
        chosen = list(np.random.choice(pool_ids, size=k, replace=False, p=probs))
        return chosen
    
    # ------------------------------------------------------------------
    # Placeholder encoder (to be replaced later)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Master helper – turns the whole static-metadata dict into ONE vec
    # ------------------------------------------------------------------

    @staticmethod
    def encode_predict(client_dicts: List[Dict[str, Any]]
                    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build the training-/inference-matrix for **gθ**.

        Parameters
        ----------
        client_dicts : List[Dict]
            Each item is
            {
                "client_id"        : <int>,
                "dynamic_metadata" : {"rates": …, "availabilities": …, "batteryLevels": …},
                "static_metadata"  : {... all static keys ...}
            }

        Returns
        -------
        X   : np.ndarray  shape = (N , 15 + 63) = (N , 78)   (float32)
            15  = 3×5 normalised dynamic values
            63   = length of encode_static_metadata()
        ids : List[int]   original client-ids in the same order as rows of X
        """
        if not client_dicts:
            return np.empty((0, 0), dtype=np.float32), []

        dyn_rows: List[np.ndarray] = []
        ids:      List[int]        = []

        # normalisation constants (vectorised)
        _RATE_MIN, _RATE_SPAN   = 1.0, 53.0         # 1 … 54
        _AVAIL_MIN, _AVAIL_SPAN = 35.0, 65.0        # 35 … 100
        _BATT_MIN, _BATT_SPAN   = -1.0, 100.0       # −1 … 99

        for rec in client_dicts:
            cid   = rec["client_id"]
            ids.append(cid)

            dyn = rec["dynamic_metadata"]  # three *5-long* numpy arrays
            rate_vec = (np.asarray(dyn["rates"],          dtype=np.float32) - _RATE_MIN)  / _RATE_SPAN
            avail_vec= (np.asarray(dyn["availabilities"], dtype=np.float32) - _AVAIL_MIN) / _AVAIL_SPAN
            batt_vec = (np.asarray(dyn["batteryLevels"],  dtype=np.float32) - _BATT_MIN)  / _BATT_SPAN

            dyn_feat = np.concatenate([rate_vec, avail_vec, batt_vec])   # (15,)

            stat_feat = encode_static_metadata(rec["static_metadata"])   # (S,)

            dyn_rows.append(np.concatenate([dyn_feat, stat_feat]))       # (15+S,)

        X = np.stack(dyn_rows).astype(np.float32)   # (N , 15+S)

        return X, ids
    
    # bliss/_training_selector.py  (inside the class)

    @staticmethod
    def encode_refresh(records: List[Dict[str, Any]]
                    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build the design-matrix to train **hϕ** (the refresh model).

        Each *record* contains both the *current* and the *previous* dynamic
        metadata, plus a few scalar history fields:

            {
                "client_id"            : int,
                "dynamic_metadata"     : {"rates": …, "availabilities": …, "batteryLevels": …},
                "last_dynamic_metadata": {"rates": …, "availabilities": …, "batteryLevels": …},
                "static_metadata"      : {...},
                "utility"              : float,
                "success"              : bool | int,
                "round"                : int,
                "last_utility"         : float,
                "last_success"         : bool | int,
                "last_round"           : int,
            }

        Returns
        -------
        X   : np.ndarray  shape = (N, 15 + 63 + 6) = (N, 84) (float32)
            ├─ 15  deltas of current – previous dynamic features
            ├─  63  static-metadata encoding
            └─  6  scalar history features
        ids : List[int]  client-ids matching X’s rows
        """
        if not records:
            return np.empty((0, 0), dtype=np.float32), []

        ids:  List[int]        = []
        rows: List[np.ndarray] = []

        # ----------- normalisation constants (same as encode_predict) ----------
        _RATE_MIN, _RATE_SPAN   = 1.0, 53.0      # 1 … 54
        _AVAIL_MIN, _AVAIL_SPAN = 35.0, 65.0     # 35 … 100
        _BATT_MIN, _BATT_SPAN   = -1.0, 100.0    # –1 … 99

        for rec in records:
            cid = rec["client_id"]
            ids.append(cid)

            cur_dyn  = rec["dynamic_metadata"]
            prev_dyn = rec["last_dynamic_metadata"]

            # --- helper --------------------------------------------------------
            def _norm(v, vmin, span):
                return (np.asarray(v, dtype=np.float32) - vmin) / span

            # current & previous (all length-5 vectors)
            r_now  = _norm(cur_dyn["rates"],          _RATE_MIN,  _RATE_SPAN)
            r_prev = _norm(prev_dyn["rates"],         _RATE_MIN,  _RATE_SPAN)

            a_now  = _norm(cur_dyn["availabilities"], _AVAIL_MIN, _AVAIL_SPAN)
            a_prev = _norm(prev_dyn["availabilities"],_AVAIL_MIN, _AVAIL_SPAN)

            b_now  = _norm(cur_dyn["batteryLevels"],  _BATT_MIN,  _BATT_SPAN)
            b_prev = _norm(prev_dyn["batteryLevels"], _BATT_MIN,  _BATT_SPAN)

            delta_dyn = np.concatenate([r_now - r_prev,
                                        a_now - a_prev,
                                        b_now - b_prev])          # (15,)

            # --- static features ----------------------------------------------
            static_vec = encode_static_metadata(rec["static_metadata"])

            # --- scalar history -----------------------------------------------
            last_util   = float(rec.get("last_utility",   0.0))
            last_succ   = 1.0 if rec.get("last_success") else 0.0
            last_round  = float(rec.get("last_round",     0))

            curr_util   = float(rec.get("utility",        0.0))
            curr_succ   = 1.0 if rec.get("success")      else 0.0
            curr_round  = float(rec.get("round",          0))

            hist_vec = np.asarray(
                [last_util, last_succ, last_round,
                curr_util, curr_succ, curr_round],
                dtype=np.float32
            )

            rows.append(np.concatenate([delta_dyn, static_vec, hist_vec]))

        X = np.stack(rows).astype(np.float32)   # (N , 15 + S + 6)
        return X, ids



    # ------------------------------------------------------------------
    # Main Selection – called by ClientManager
    # ------------------------------------------------------------------

    def _is_seen(self, cid: int) -> bool:
        return self.clients[cid]['seen'] > 0

    def request_clients_to_refresh_utility(self, online_ids: List[int]) -> List[int]:
        """
        Return up to `self.number_clients_to_refresh_utility` *seen* clients,
        ranked by current utility.
        """

        seen_online = [cid for cid in online_ids if self._is_seen(cid)]
        if len(seen_online) <= self.number_clients_to_refresh_utility:
            return seen_online

        # rank by utility, higher first
        ranked = sorted(seen_online,
                        key=lambda cid: self.clients[cid]['utility'],
                        reverse=True)
        return ranked[: self.number_clients_to_refresh_utility]


    def request_clients_to_predict_utility(self, online_ids: List[int]) -> List[int]:
        """
        Uniformly sample up to `self.number_clients_to_predict_utility`
        *unseen* clients from those currently online.
        """
        unseen_online = [cid for cid in online_ids if not self._is_seen(cid)]
        k = min(self.number_clients_to_predict_utility, len(unseen_online))
        if k == 0:
            return []
        return self.rng.sample(unseen_online, k)

    def send_clients_to_predict(self, client_metadata: Dict[str, Any]):
        self.clients_to_predict.append(client_metadata)

    def send_clients_to_refresh(self, client_metadata: Dict[str, Any]):
        self.clients_to_refresh.append(client_metadata) 


    def select_participant(self, num_of_clients: int) -> List[int]:
        """Return ``num_of_clients`` client IDs with the highest *predicted* utility."""

        # ------------------------------------------------------------------
        # 1 ▸ TRAIN **gθ** – map (static + current dynamic) → utility
        #     Training data: any client we have *already* seen at least once.
        # ------------------------------------------------------------------
        seen_once_ids = [cid for cid, info in self.clients.items() if info["seen"] > 0]
        train_predict_ids = self._weighted_sample(
            seen_once_ids,
            min(self.amount_clients_predict_train_set, len(seen_once_ids)),
        )

        if train_predict_ids:
            train_dicts = [
                {
                    "client_id": cid,
                    "dynamic_metadata": self.clients[cid]["dynamic_metadata"],
                    "static_metadata": self.clients[cid]["static_metadata"],
                }
                for cid in train_predict_ids
            ]
            try:
                X_train, _ = self.encode_predict(train_dicts)
                y_train = np.array([self.clients[cid]["utility"] for cid in train_predict_ids])
                if X_train.size > 0 and not np.allclose(y_train, 0):
                    self._g_w, self._g_b = _linreg_fit(X_train, y_train)
            except NotImplementedError:
                logging.warning("[Bliss] encode() not implemented – gθ not updated")

        # ------------------------------------------------------------------
        # 2 ▸ TRAIN **hϕ** – refresh model.  Need at least 2 observations / client.
        # ------------------------------------------------------------------
        seen_twice_ids = [cid for cid, info in self.clients.items() if info["seen"] > 1]
        train_refresh_ids = self._weighted_sample(
            seen_twice_ids,
            min(self.amount_clients_refresh_train_set, len(seen_twice_ids)),
        )

        if train_refresh_ids:
            enriched: List[Dict[str, Any]] = []
            for cid in train_refresh_ids:
                base = self.clients[cid]
                enriched.append(
                    {
                        "client_id": cid,
                        "dynamic_metadata": base["dynamic_metadata"],
                        "last_dynamic_metadata": base["last_dynamic_metadata"],
                        "static_metadata": base["static_metadata"],
                        "utility": base["utility"],
                        "success": base["success"],
                        "round": base["round"],
                        "last_utility": base["last_utility"],
                        "last_success": base["last_success"],
                        "last_round": base["last_round"],
                    }
                )
            try:
                X_train_r, _ = self.encode_refresh(enriched)
                y_train_r = np.array([self.clients[cid]["utility"] for cid in train_refresh_ids])
                if X_train_r.size > 0 and not np.allclose(y_train_r, 0):
                    self._h_w, self._h_b = _linreg_fit(X_train_r, y_train_r)
            except NotImplementedError:
                logging.warning("[Bliss] encode() not implemented – hϕ not updated")

        # ------------------------------------------------------------------
        # 3 ▸ PREDICT utilities for the online candidates passed via ClientManager
        # ------------------------------------------------------------------
        predictions: List[Tuple[int, float]] = []

        # --- (a) Unseen online clients → gθ ---------------------------------
        if self.clients_to_predict:
            # attach the per-client static features we stored at registration
            enriched_predict: List[Dict[str, Any]] = []
            for d in self.clients_to_predict:
                cid = d["client_id"]
                enriched_predict.append(
                    {
                        **d,                                   # dynamic metadata already present
                        "static_metadata": self.clients[cid]["static_metadata"],
                    }
                )

            try:
                X_pred, ids_pred = self.encode_predict(enriched_predict)
                if self._g_w is not None and X_pred.size > 0:
                    util_hat = _linreg_predict(X_pred, self._g_w, self._g_b)
                else:
                    util_hat = np.zeros(len(ids_pred), dtype=float)
                predictions.extend(zip(ids_pred, util_hat.tolist()))
            except NotImplementedError:
                logging.warning("[Bliss] encode() not implemented – assigning zero utility to unseen predictions")
                predictions.extend((d["client_id"], 0.0) for d in enriched_predict)


        # --- (b) Seen online clients to refresh → hϕ ------------------------
        if self.clients_to_refresh:
            enriched_refresh: List[Dict[str, Any]] = []
            for d in self.clients_to_refresh:
                cid = d["client_id"]
                base = self.clients[cid]
                enriched_refresh.append(
                    {
                        **d,  # dynamic data already present
                        "static_metadata": base["static_metadata"],
                        "last_dynamic_metadata": base["dynamic_metadata"],
                        "last_utility": base["utility"],
                        "last_success": base["success"],
                        "last_round": base["round"],
                    }
                )
            try:
                X_ref, ids_ref = self.encode_refresh(enriched_refresh)
                if self._h_w is not None and X_ref.size > 0:
                    util_hat_r = _linreg_predict(X_ref, self._h_w, self._h_b)
                else:
                    util_hat_r = np.zeros(len(ids_ref))
                predictions.extend(list(zip(ids_ref, util_hat_r.tolist())))
            except NotImplementedError:
                logging.warning("[Bliss] encode() not implemented – assigning zero utility to refresh predictions")
                predictions.extend([(d["client_id"], 0.0) for d in enriched_refresh])

        # ------------------------------------------------------------------
        # 4 ▸ Handle edge cases & pick top‑K
        # ------------------------------------------------------------------
        picked = []
        if not predictions:
            logging.warning("[Bliss] … – fallback to random")
            picked = self.rng.sample(list(self.clients.keys()), k=min(num_of_clients, len(self.clients)))
        else:
            predictions.sort(key=lambda t: t[1], reverse=True)
            picked = [cid for cid, _ in predictions[:num_of_clients]]


        # Pad if needed (should be rare)
        if len(picked) < num_of_clients:
            logging.info(f"[Bliss] only {len(picked)} clients out of the requested {num_of_clients}")

        # ------------------------------------------------------------------
        # 5 ▸ Book‑keeping & cleanup
        # ------------------------------------------------------------------
        for cid in picked:
            self.clients[cid]["seen"] += 1
            self.clients[cid]["round"] = self.round

        self.round += 1
        self.clients_to_predict.clear()
        self.clients_to_refresh.clear()

        return picked

    # ------------------------------------------------------------------
    # Utility helpers (some remain stubbed)
    # ------------------------------------------------------------------

    def calculateSumUtil(self, clientList: Sequence[int]) -> float:  # noqa: N802 – keep Oort naming
        return float(sum(self.clients[c]['utility'] for c in clientList if c in self.clients))

    def get_median_reward(self) -> float:
        utils = [c['utility'] for c in self.clients.values()]
        if not utils:
            return 0.0
        return float(np.median(utils))

    def getAllMetrics(self):  # noqa: N802 – keep Oort naming
        """
        Return a rich snapshot of the current state.

        Keys
        ----
        round                – index of *next* round to be scheduled
        seen                 – number of clients with at least one successful run
        unseen               – total registered − seen
        avg_util / min_util / max_util
                             – statistics over *all* clients that finished the
                               most-recent round (including stragglers)
        avg_util_no_strag / …
                             – same, but only for clients whose last run
                               succeeded (`success == True`)
        stragglers           – #clients that participated last round but failed
        """
        # ------------- basic counters ------------------------------------
        seen_cnt   = sum(1 for c in self.clients.values() if c["seen"] > 0)
        unseen_cnt = len(self.clients) - seen_cnt

        # ------------- stats for the most-recent completed round ---------
        last_round = self.round - 1  # because self.round was incremented after scheduling

        utils_last_round = [
            c["utility"] for c in self.clients.values()
            if c["round"] == last_round
        ]
        succ_utils_last_round = [
            c["utility"] for c in self.clients.values()
            if c["round"] == last_round and c["success"]
        ]

        def _summ(stats: list[float]) -> tuple[float, float, float]:
            if not stats:
                return (0.0, 0.0, 0.0)
            return (float(np.mean(stats)), float(np.min(stats)), float(np.max(stats)))

        avg_u,  min_u,  max_u  = _summ(utils_last_round)
        avg_ns, min_ns, max_ns = _summ(succ_utils_last_round)

        stragglers = len(utils_last_round) - len(succ_utils_last_round)

        return {
            "round":                 int(self.round),
            "seen":                  int(seen_cnt),
            "unseen":                int(unseen_cnt),
            "avg_util":              avg_u,
            "min_util":              min_u,
            "max_util":              max_u,
            "avg_util_no_strag":     avg_ns,
            "min_util_no_strag":     min_ns,
            "max_util_no_strag":     max_ns,
            "stragglers":            int(stragglers),
        }







# -----------------------------------------------------------------------------
# Testing selector – shape compatible with Oort.  *Mostly placeholder*
# -----------------------------------------------------------------------------

class _testing_selector:  # noqa: D401 – keep Oort naming
    """Bliss testing‑phase participant selector (stub)."""

    def __init__(
        self,
        data_distribution: Optional[Dict[Any, Any]] = None,
        client_info: Optional[Dict[int, Sequence[float]]] = None,
        model_size: Optional[int] = None,
    ) -> None:
        self.data_distribution = data_distribution or {}
        self.client_info = client_info or {}
        self.model_size = model_size or 0
        self.client_idx_list = list(self.client_info.keys()) if self.client_info else []
        logging.debug("[Bliss/TestSel] initialised with %d clients", len(self.client_info))

    # ------- API stubs --------------------------------------------------

    def select_by_deviation(
        self,
        dev_target: float,
        range_of_capacity: Tuple[float, float],
        total_num_clients: int,
        confidence: float = 0.8,
        overcommit: float = 1.1,
    ) -> int:  # noqa: D401 – keep Oort signature
        # Very simple Hoeffding bound clone
        low, high = range_of_capacity
        rng = high - low
        m = int((rng**2) * np.log(2 / (1 - confidence)) / (2 * (dev_target**2)))
        return int(np.ceil(m * overcommit))

    def select_by_category(
        self,
        request_list: List[Dict[str, Any]],
        max_num_clients: Optional[int] = None,
        greedy_heuristic: bool = True,
    ) -> Tuple[List[int], float, float]:
        raise NotImplementedError("Category‑aware testing not implemented for Bliss yet.")


# -----------------------------------------------------------------------------
# Internal helper – extremely light linear regressor for drift/prediction
# -----------------------------------------------------------------------------

def _linreg_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (weights, bias) solving y ≈ Xw + b via least‑squares."""
    # Add bias term
    X_ = np.hstack([X, np.ones((X.shape[0], 1))])
    w, *_ = np.linalg.lstsq(X_, y, rcond=None)
    return w[:-1], w[-1]


def _linreg_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b

