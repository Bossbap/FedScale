import logging
import copy
import pickle
from random import Random
from typing import List, Tuple, Dict, Optional
import numpy as np
import math
from pathlib import Path

from fedscale.cloud.internal.client_metadata import ClientMetadata


class ClientManager:

    def __init__(self, mode, args, sample_seed=233):
        self.client_metadata = {}
        self.client_on_hosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucb_sampler = None

        if self.mode == 'oort':
            from thirdparty.oort.oort import create_training_selector
            self.ucb_sampler = create_training_selector(args=args, sample_seed=sample_seed)

        if self.mode == 'pyramidfl':
            from thirdparty.oort.oort import create_pyramid_training_selector
            self.pyr_sampler = create_pyramid_training_selector(args=args, sample_seed=sample_seed)
            self._pyr_overrides: Dict[int, Dict[str, float | int]] = {}
            self._round_start_time: float = 0.0

        if self.mode == 'bliss':
            from thirdparty.bliss.bliss import create_training_selector
            self.bliss_sampler = create_training_selector(args=args, sample_seed=sample_seed)


        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self._np_rng = np.random.default_rng(sample_seed)
        self._client_seed_base = int(sample_seed) % (2**32)
        self.count = 0
        self.feasible_samples = 0
        self.args = args

        with open(args.clients_file, 'rb') as fin:
            self.clients = pickle.load(fin)
        # Ensure deterministic order for modulo mapping even if the pickle is an OrderedDict
        self.clients_keys = sorted(self.clients.keys())
        self._base_pool_size = len(self.clients_keys)
        self._cluster_ranks = self._load_cluster_ranks()
        # Noise hyperparams (kept local to ClientManager; can be surfaced to args if desired)
        self._noise_std = 0.07   # truncated N(0, 0.07)
        self._noise_lo  = -1.0   # truncate to [-1, 1]
        self._noise_hi  =  1.0
        # Feature ranges for clipping/denorm
        self._ranges = {
            "rate":         (1.0, 54.0),
            "availability": (35.0, 100.0),
            "batteryLevel": (-1.0, 99.0),
        }

        self._last_online_log_clock: Optional[float] = None

    def _load_cluster_ranks(self) -> Dict[int, int]:
        """Load {client_id -> cluster_rank} mapping from the canonical dataset."""
        cluster_map: Dict[int, int] = {}
        cluster_file = Path("benchmark/dataset/data/clients.pkl")
        if not cluster_file.exists():
            logging.warning(
                "[ClientManager] cluster-rank file missing at %s – defaulting to rank 0",
                cluster_file,
            )
            return cluster_map
        try:
            with cluster_file.open("rb") as fin:
                data = pickle.load(fin)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning(
                "[ClientManager] unable to load cluster ranks from %s: %s",
                cluster_file,
                exc,
            )
            return cluster_map

        for key, rec in data.items():
            rank = rec.get("cluster_rank")
            if rank is None:
                continue
            try:
                rank_val = int(rank)
            except (TypeError, ValueError):
                continue
            cluster_map[int(key)] = rank_val
            rec_id = rec.get("id")
            if rec_id is not None:
                cluster_map[int(rec_id)] = rank_val
        return cluster_map

    def register_client(self, host_id: int, client_id: int, size: int) -> None:
        """Register client information to the client manager.

        Args:
            host_id (int): executor Id.
            client_id (int): client Id.
            size (int): number of samples on this client.
            duration (float): (ignored for Oort init) execution latency; we force 0.
        """
        if client_id in self.client_metadata:
            return

        # --- Map real client_id to a base profile (wrap if needed), then add noise if wrapped ---
        if client_id in self.clients:
            cd = self.clients[client_id]
        else:
            base_key = self.clients_keys[client_id % self._base_pool_size]
            base_cd  = self.clients[base_key]
            cd = self._synthesise_client(base_cd, client_id)
            
        client_seed = (self._client_seed_base + int(client_id)) % (2 ** 32)

        self.client_metadata[client_id] = ClientMetadata(
            host_id=host_id,
            client_id=client_id,
            size=size,
            cpu_flops=cd['CPU_FLOPS'],
            gpu_flops=cd['GPU_FLOPS'],
            timestamps_livelab=cd['timestamps-livelab'],
            rate=cd['rate'],
            timestamps_carat=cd['timestamps-carat'],
            availability=cd['availability'],
            batteryLevel=cd['batteryLevel'],
            active=cd['active'],
            inactive=cd['inactive'],
            peak_throughput=cd['peak_throughput'],
            rng_seed=client_seed,
        )

        # Admit client iff size is within bounds
        if self.filter_less <= size <= self.filter_more:
            self.feasibleClients.append(client_id)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {
                    'reward': 0.0,   # statistical utility prior = 0
                    'duration': 0.0, # system utility prior = 0
                }
                self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
            
            elif self.mode == "pyramidfl":
                feedbacks = {
                    'reward': 0.0,
                    'duration': 0.0,
                    'gsize': 0.0,
                    't_comp': None,
                    't_total': None,
                }
                self.pyr_sampler.register_client(client_id, feedbacks=feedbacks)

            elif self.mode == "bliss":
                cluster_rank = cd.get('cluster_rank')
                if cluster_rank is None:
                    cluster_rank = self._cluster_ranks.get(client_id)
                if cluster_rank is None:
                    base_id = cd.get('id')
                    if base_id is not None:
                        cluster_rank = self._cluster_ranks.get(int(base_id), None)
                try:
                    cluster_rank_val = int(cluster_rank)
                except (TypeError, ValueError):
                    cluster_rank_val = 0

                feedbacks = {
                    'metadata': {
                        'cpu_flops': cd['CPU_FLOPS'],
                        'gpu_flops': cd['GPU_FLOPS'],
                        'peak_throughput': cd['peak_throughput'],
                        'cluster_rank': cluster_rank_val,
                    }
                }
                self.bliss_sampler.register_client(client_id, feedbacks)
        else:
            del self.client_metadata[client_id]


    def select_participants(self, num_of_clients: int, cur_time: float = 0) -> List[int]:
        """Select participating clients for current execution task.

        We always route Oort selection through the sampler (no special casing for round=1),
        because the sampler now handles the unexplored/explored split itself.
        """
        self.count += 1

        clients_online = self.getOnlineClients(cur_time)
        clients_online_set = set(clients_online)

        if self.mode == "oort":
            return self.ucb_sampler.select_participant(
                num_of_clients, feasible_clients=clients_online_set
            )

        if self.mode == "pyramidfl":
            self._round_start_time = cur_time
            picked = self.pyr_sampler.select_participant(num_of_clients, feasible_clients=clients_online_set)
            self._pyr_overrides = self.pyr_sampler.get_overrides()
            return picked

        elif self.mode == "bliss":
            clients_to_predict_utility = self.bliss_sampler.request_clients_to_predict_utility(clients_online)
            clients_to_refresh_utility = self.bliss_sampler.request_clients_to_refresh_utility(clients_online)

            self.send_bliss_metadata(clients_to_predict_utility, cur_time, self.bliss_sampler.send_clients_to_predict)
            self.send_bliss_metadata(clients_to_refresh_utility, cur_time, self.bliss_sampler.send_clients_to_refresh)

            pickled_clients = self.bliss_sampler.select_participant(num_of_clients)
            self.send_bliss_metadata(pickled_clients, cur_time, self.bliss_sampler.update_client_metadata_pre_training)
            return pickled_clients

        else:
            self.rng.shuffle(clients_online)
            return clients_online[:num_of_clients]


    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, client_id):
        return self.client_metadata[client_id]

    def registerDuration(self, client_id, duration):
        if self.mode == "oort":
            self.ucb_sampler.update_duration(client_id, duration)

        elif self.mode == "pyramidfl":
            self.pyr_sampler.update_duration(client_id, duration)

        meta = self.client_metadata.get(client_id)
        if meta is not None:
            meta.last_duration = duration

    def get_completion_time(self, client_id, cur_time, batch_size, local_steps, model_size, model_amount_parameters):

        client_completion_time =  self.client_metadata[client_id].get_completion_time(
            cur_time=cur_time,
            batch_size=batch_size,
            local_steps=local_steps,
            model_size=model_size,
            model_amount_parameters=model_amount_parameters,
            clock_factor=self.args.clock_factor
        )

        return client_completion_time


    def registerSpeed(self, host_id, client_id, speed):
        uniqueId = self.getUniqueId(host_id, client_id)
        self.client_metadata[uniqueId].speed = speed

    def registerScore(self, client_id, reward, time_stamp=0, duration=1., success=True):
        self.register_feedback(client_id, reward, time_stamp=time_stamp, duration=duration, success=success)

    def register_feedback(self, client_id: int, reward: float, time_stamp: float = 0,
                          duration: float = 1., success: bool = True, **kwargs) -> None:

        """Collect client execution feedbacks of last round.

        Args:
            client_id (int): client Id.
            reward (float): execution utilities (processed feedbacks).
            time_stamp (float): current wall clock time.
            duration (float): system execution duration.
            success (bool): whether this client runs successfully.

        """
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }

            self.ucb_sampler.update_client_util(client_id, feedbacks=feedbacks)

        elif self.mode == "pyramidfl":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': bool(success),
                'time_stamp': time_stamp,
                'gsize': kwargs.get('gsize', None),
                't_comp': kwargs.get('t_comp', None),
                't_total': kwargs.get('t_total', None),
                # record how many local iterations actually ran on the client
                'steps': kwargs.get('steps', None),
            }
            self.pyr_sampler.update_client_util(client_id, feedbacks=feedbacks)

        elif self.mode == "bliss":
            stored_reward = float(reward) if success else 0.0

            if success and not getattr(self.args, "adaptive_training", False):
                t_budget = float(getattr(self.args, "t_budget", 0.0) or 0.0)
                round_penalty = float(getattr(self.args, "round_penalty", 0.0) or 0.0)
                dur = float(duration) if duration is not None else None

                if t_budget > 0.0 and dur is not None:
                    if dur > t_budget:
                        penalty = (dur / t_budget) ** round_penalty
                    else:
                        penalty = 1.0
                    stored_reward *= penalty

            feedbacks = {
                'reward': stored_reward,
                'success': success
            }
            self.bliss_sampler.update_client_metadata_post_training(client_id, feedbacks)

    def registerClientScore(self, client_id, reward):
        self.client_metadata[self.getUniqueId(0, client_id)].register_reward(reward)

    def get_score(self, host_id, client_id):
        uniqueId = self.getUniqueId(host_id, client_id)
        return self.client_metadata[uniqueId].get_score()

    def getClientsInfo(self):
        clientInfo = {}
        for i, client_id in enumerate(self.client_metadata.keys()):
            client = self.client_metadata[client_id]
            clientInfo[client.client_id] = client.distance
        return clientInfo

    def next_client_id_to_run(self, host_id):
        init_id = host_id - 1
        lenPossible = len(self.feasibleClients)

        while True:
            client_id = str(self.feasibleClients[init_id])
            csize = self.client_metadata[client_id].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(client_id)

            init_id = max(
                0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def clientSampler(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def clientOnHost(self, client_ids, host_id):
        self.client_on_hosts[host_id] = client_ids

    def getCurrentclient_ids(self, host_id):
        return self.client_on_hosts[host_id]

    def getClientLenOnHost(self, host_id):
        return len(self.client_on_hosts[host_id])

    def getClientSize(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def getSampleRatio(self, client_id, host_id, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.client_on_hosts.keys():
                for client in self.client_on_hosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.client_metadata[uniqueId].size

            # 1./len(self.client_on_hosts.keys())
            return float(self.client_metadata[self.getUniqueId(host_id, client_id)].size) / float(totalSampleInTraining)
        else:
            for key in self.client_on_hosts.keys():
                totalSampleInTraining += len(self.client_on_hosts[key])

            return 1. / totalSampleInTraining

    def getOnlineClients(self, cur_time):
        clients_online = [client_id for client_id in self.feasibleClients if self.client_metadata[client_id].is_active(cur_time)]
        if (
            self._last_online_log_clock is None
            or abs(cur_time - self._last_online_log_clock) > 1e-6
        ):
            logging.info(
                f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, "
                f"{len(self.feasibleClients) - len(clients_online)} clients offline"
            )
            self._last_online_log_clock = cur_time

        return clients_online

    def isClientActive(self, client_id, cur_time):
        return self.client_metadata[client_id].is_active(cur_time)
    
    def isClientActiveThroughout(self, client_id: int, start_time: float, end_time: float) -> bool:
        """Return True if the client stays active for the entire interval [start_time, end_time).

        Uses the client's 48h cyclic active/inactive schedule and checks coverage across
        any number of wrapped cycles.
        """
        if end_time <= start_time:
            return True

        meta = self.client_metadata[client_id]
        T = 48 * 3600.0

        # Build active intervals over [0, T)
        act_intervals = self._active_intervals(meta.active, meta.inactive)
        if not act_intervals:
            return False

        def covered_in_cycle(seg_s: float, seg_e: float) -> bool:
            """Check if [seg_s, seg_e) within one cycle [0,T) is fully covered by union of act_intervals."""
            pos = seg_s
            for (a, b) in act_intervals:
                if b <= pos:
                    continue
                if a > pos:
                    # gap before next active interval
                    return False
                # a <= pos < b  → extend coverage
                pos = min(seg_e, b)
                if pos >= seg_e:
                    return True
            return pos >= seg_e

        t = start_time
        while t < end_time:
            r = t % T
            cycle_end = t - r + T
            sub_end = min(end_time, cycle_end)
            seg_len = sub_end - t
            # segment normalized into [0,T)
            seg_s = r
            seg_e = r + seg_len
            if not covered_in_cycle(seg_s, seg_e):
                return False
            t = sub_end
        return True
    

    # ──────────────────────────────────────────────────────────────
    #  Bliss
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def extract_last5_windows(
            norm_t: float,
            timestamps_livelab: np.ndarray,
            rate: np.ndarray,
            timestamps_carat: np.ndarray,
            availability: np.ndarray,
            batteryLevel: np.ndarray,
            active,
            inactive
        ):
        """
        norm_t               -- current time in [0, 48*3600)
        timestamps_livelab   -- 1-D np.array (ascending, wrapped @ 48 h)
        rate                 -- 1-D np.array aligned with timestamps_livelab
        timestamps_carat     -- 1-D np.array (ascending, wrapped @ 48 h)
        availability         -- 1-D np.array aligned with timestamps_carat
        batteryLevel         -- 1-D np.array aligned with timestamps_carat
        active               -- array defining client activity intervals
        inactive             -- array defining client inactivity intervals
        --------------------------------------------------------------------
        returns  rates[5], avail[5], battLvl[5]  (newest at index 4)
        """

        def _prev_index(idx, n):
            """Circular index stepping backwards once in a list of length n."""
            return (idx - 1) % n
        
        def is_active(active, inactive, cur_time):
            """
            Determines whether the client is active at the given simulation time.

            Args:
                cur_time (int or float): Current simulation time in seconds.

            Returns:
                bool: True if client is active, False otherwise.
            """
            T = 48 * 3600
            t = cur_time % T

            # Merge the two sorted lists, tag each timestamp with the phase it *starts*
            boundaries = sorted(
                [(ts, 'a') for ts in active] +
                [(ts, 'i') for ts in inactive]
            )

            # Initial phase
            phase = 'a' if (active and active[0] == 0) else 'i'

            # Walk through boundaries and flip the phase whenever we pass one
            for ts, _ in boundaries[1:]:  # skip the initial 0 entry
                if t < ts:
                    break
                phase = 'i' if phase == 'a' else 'a'

            return phase == 'a'

        def _fill_series(ts, vals, active, inactive):
            """internal: build one 5-value history list for a single series"""
            n = len(ts)

            # ----- find latest index <= norm_t -----
            if norm_t < ts[0]:
                idx = n - 1                       # wrap around
            else:
                idx = np.searchsorted(ts, norm_t, side='right') - 1

            out = np.empty(5, dtype=vals.dtype)
            out[4] = vals[idx]                   # most recent observation

            last_good = out[4]                   # last value actually kept

            # walk four more steps backwards
            for k in range(3, -1, -1):           # fill slots 3,2,1,0
                prev_idx = _prev_index(idx, n)
                t_new   = ts[prev_idx]
                t_old   = ts[idx]

                # mid-point to test activity
                mid_t = (t_old + t_new) / 2.0
                # wrap midpoint if we crossed 0 on the circular time line
                if t_old < t_new:                # crossed 0 boundary
                    mid_t = (mid_t + 24*3600) % (48*3600)

                if is_active(active, inactive, mid_t):          # OK – keep real value
                    last_good = vals[prev_idx]
                # else: keep last_good (i.e. duplicate)

                out[k] = last_good
                idx = prev_idx                   # move the cursor

            return out

        rates          = _fill_series(timestamps_livelab, rate, active, inactive)
        availabilities = _fill_series(timestamps_carat, availability, active, inactive)
        batteryLevels  = _fill_series(timestamps_carat, batteryLevel, active, inactive)

        return rates, availabilities, batteryLevels

    def send_bliss_metadata(self, clients: list[int], cur_time, update_fn):

        for client_id in clients:
            client_metadata = self.client_metadata[client_id]

            timestamps_livelab = client_metadata.timestamps_livelab
            rate = client_metadata.rate

            timestamps_carat = client_metadata.timestamps_carat
            availability = client_metadata.availability
            batteryLevel = client_metadata.batteryLevel

            active = client_metadata.active
            inactive = client_metadata.inactive

            norm_t = cur_time % (48 * 3600)

            rates, availabilities, batteryLevels = self.extract_last5_windows(norm_t, timestamps_livelab, rate, timestamps_carat, availability, batteryLevel, active, inactive)

            update_fn(
                    {
                        'client_id': client_id,
                        'dynamic_metadata':
                        {
                            'rates': rates,
                            'availabilities': availabilities,
                            'batteryLevels': batteryLevels
                        }
                    }
                )
            
    # ──────────────────────────────────────────────────────────────
    #  PyramidFL
    # ──────────────────────────────────────────────────────────────
            

    def get_pyramidfl_conf(self, client_id: int) -> Optional[dict]:
        """Return per-client overrides computed by PyramidFL for the CURRENT round."""
        return self._pyr_overrides.get(client_id, None)

    def get_times_pyramid(
        self,
        client_id: int,
        cur_time: float,
        batch_size: int,
        local_steps: int,
        model_size: int,
        model_amount_parameters: int,
        dropout_p: float,
    ) -> Tuple[float, float]:
        """Return (t_comp, t_total) for PyramidFL with given overrides."""
        return self.client_metadata[client_id].get_times_with_dropout(
            cur_time=cur_time,
            batch_size=batch_size,
            local_steps=local_steps,
            model_size=model_size,
            model_amount_parameters=model_amount_parameters,
            reduction_factor=0.5,
            dropout_p=dropout_p,
            clock_factor=self.args.clock_factor,
        )


    # ──────────────────────────────────────────────────────────────
    #  Utils
    # ──────────────────────────────────────────────────────────────

    def resampleClients(self, num_of_clients, cur_time=0):
        return self.select_participants(num_of_clients, cur_time)

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucb_sampler.getAllMetrics()
        elif self.mode == "pyramidfl":
            # Expose PyramidFL internal metrics (same structure as Oort's totalArms)
            return self.pyr_sampler.getAllMetrics()
        elif self.mode == "bliss":
            return self.bliss_sampler.getAllMetrics()

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}
    
    


    # ──────────────────────────────────────────────────────────────
    #  Synthetic client helpers (wrap + noisy dynamics)
    # ──────────────────────────────────────────────────────────────
    def _trunc_gauss(self, sigma: float) -> float:
        """Sample ε ~ N(0, σ) truncated to [-1, 1]."""
        while True:
            eps = self._np_rng.normal(loc=0.0, scale=sigma)
            if self._noise_lo <= eps <= self._noise_hi:
                return float(eps)

    @staticmethod
    def _active_intervals(active: list[int], inactive: list[int]) -> list[tuple[float, float]]:
        """Return [(start,end), ...] active intervals over [0, 48h). Handles wrap-around."""
        T = 48 * 3600.0
        # Merge boundaries tagged with the phase that begins at the timestamp
        boundaries = sorted([(ts, 'a') for ts in active] + [(ts, 'i') for ts in inactive])
        if not boundaries:
            return []  # no information; treat as always inactive
        # Initial phase at t=0
        phase = 'a' if (active and active[0] == 0) else 'i'
        intervals = []
        last_ts = 0.0
        for ts, _ in boundaries[1:]:
            # [last_ts, ts) carries the previous phase
            if phase == 'a':
                intervals.append((last_ts, float(ts)))
            phase = 'i' if phase == 'a' else 'a'
            last_ts = float(ts)
        # Final tail to T
        if phase == 'a':
            intervals.append((last_ts, T))
        # If first boundary not at 0, we also have a head segment [0, boundaries[0].ts)
        first_ts = float(boundaries[0][0])
        phase0 = 'a' if (active and active[0] == 0) else 'i'
        if first_ts > 0 and phase0 == 'a':
            intervals.insert(0, (0.0, first_ts))
        # Normalize / split any wrap-around (shouldn’t exist after above logic, but keep safe)
        out = []
        for s, e in intervals:
            if e >= s:
                out.append((s, e))
            else:
                out.append((s, T))
                out.append((0.0, e))
        return out

    @staticmethod
    def _ts_in_any_interval(ts: float, intervals: list[tuple[float, float]]) -> bool:
        for s, e in intervals:
            if s <= ts < e:
                return True
        return False

    def _apply_activity_noise(
        self,
        timestamps: list[int],
        values: list[float],
        act_intervals: list[tuple[float, float]],
        feature_name: str,
    ) -> list[float]:
        """Add a single ε (per *active interval*) to all values whose segment starts inside that interval.
        Values are normalized to [0,1], shifted by ε, clipped to [0,1], then denormalized back
        and clipped to the feature’s real range.
        """
        lo, hi = self._ranges[feature_name]
        T = 48 * 3600.0
        ts_arr = np.asarray(timestamps, dtype=np.float32)
        vals   = np.asarray(values, dtype=np.float32)
        out    = vals.copy()

        if len(ts_arr) == 0 or len(vals) == 0:
            return values

        # Build a lookup from interval -> indices whose segment starts in it
        # For each active interval, sample one ε and apply to all those indices.
        for (s, e) in act_intervals:
            # indices i where ts_i ∈ [s, e)
            mask = (ts_arr >= s) & (ts_arr < e)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            eps = self._trunc_gauss(self._noise_std)
            # normalize, shift, clip, denormalize, clip
            v = out[idxs]
            v_norm = (v - lo) / (hi - lo)
            v_norm = np.clip(v_norm + eps, 0.0, 1.0)
            v_new  = lo + v_norm * (hi - lo)
            out[idxs] = np.clip(v_new, lo, hi)

        return out.tolist()

    def _synthesise_client(self, base_cd: dict, new_client_id: int) -> dict:
        """Create a synthetic client by copying base_cd and applying activity-period noise to dynamics."""
        cd = copy.deepcopy(base_cd)
        # Activity schedule (kept identical)
        act = cd.get('active', [])
        ina = cd.get('inactive', [])
        act_intervals = self._active_intervals(act, ina)

        # Apply noise to the three dynamic series *during active periods only*
        # rate -> timestamps-livelab
        cd['rate'] = self._apply_activity_noise(
            timestamps=cd['timestamps-livelab'],
            values=cd['rate'],
            act_intervals=act_intervals,
            feature_name='rate',
        )
        # availability & batteryLevel -> timestamps-carat
        cd['availability'] = self._apply_activity_noise(
            timestamps=cd['timestamps-carat'],
            values=cd['availability'],
            act_intervals=act_intervals,
            feature_name='availability',
        )
        cd['batteryLevel'] = self._apply_activity_noise(
            timestamps=cd['timestamps-carat'],
            values=cd['batteryLevel'],
            act_intervals=act_intervals,
            feature_name='batteryLevel',
        )
        return cd
    

    def get_pacer_state(self):
        if self.mode == "bliss" and hasattr(self, "bliss_sampler"):
            return self.bliss_sampler.get_pacer_state()
        if self.mode == "oort" and hasattr(self, "ucb_sampler"):
            return self.ucb_sampler.get_pacer_state()
        if self.mode == "pyramidfl" and hasattr(self, "pyr_sampler"):
            return self.pyr_sampler.get_pacer_state()
        return {"algo": self.mode, "note": "no pacer state"}

    # Determinism diagnostics: expose sampler RNG heads
    # ------------------------------------------------------------------
    #  Serialisation helpers
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Return a serialisable snapshot of the ClientManager state."""
        if self.mode == "oort" and self.ucb_sampler is not None:
            sampler_state = self.ucb_sampler.get_state()
        elif self.mode == "pyramidfl":
            sampler_state = self.pyr_sampler.get_state()
        elif self.mode == "bliss":
            sampler_state = self.bliss_sampler.get_state()
        else:
            sampler_state = None

        metadata_state = {
            cid: meta.state_dict() for cid, meta in self.client_metadata.items()
        }

        return {
            "mode": self.mode,
            "rng_state": self.rng.getstate(),
            "np_rng_state": self._np_rng.bit_generator.state,
            "feasibleClients": list(self.feasibleClients),
            "feasible_samples": self.feasible_samples,
            "count": self.count,
            "client_metadata": metadata_state,
            "sampler_state": sampler_state,
            "_pyr_overrides": getattr(self, "_pyr_overrides", {}),
            "_round_start_time": getattr(self, "_round_start_time", 0.0),
        }

    def load_state(self, state: dict) -> None:
        """Restore ClientManager state from `get_state` output."""
        if not state:
            return

        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)
        np_state = state.get("np_rng_state")
        if np_state is not None:
            self._np_rng = np.random.default_rng()
            self._np_rng.bit_generator.state = np_state

        self.feasibleClients = list(state.get("feasibleClients", []))
        self.feasible_samples = state.get("feasible_samples", self.feasible_samples)
        self.count = state.get("count", self.count)

        self.client_metadata = {}
        meta_blob = state.get("client_metadata", {})
        for cid, mstate in meta_blob.items():
            cm = ClientMetadata.from_state(mstate)
            self.client_metadata[int(cid)] = cm

        sampler_state = state.get("sampler_state")
        if sampler_state:
            if self.mode == "oort" and self.ucb_sampler is not None:
                self.ucb_sampler.load_state(sampler_state)
            elif self.mode == "pyramidfl":
                self.pyr_sampler.load_state(sampler_state)
            elif self.mode == "bliss":
                self.bliss_sampler.load_state(sampler_state)

        if self.mode == "pyramidfl":
            self._pyr_overrides = state.get("_pyr_overrides", {})
            self._round_start_time = state.get("_round_start_time", 0.0)
