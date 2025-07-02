import numpy as np
from typing import Callable
from fedscale.cloud.fllibs import *


class ClientMetadata:
    """
    Contains the server-side metadata for a single client,
    including static device capacities and dynamic time-varying traces.
    """

    def __init__(
        self,
        host_id: int,
        client_id: int,
        size: int,
        # Static compute capacities
        cpu_flops: float,
        gpu_flops: float,
        # Dynamic traces: livelab (network) 
        timestamps_livelab: list[int],
        rate: list[int],
        # Dynamic traces: carat (compute/availability)
        timestamps_carat: list[int],
        availability: list[int],
        batteryLevel: list[int],
        # activity traces
        active: list[int],
        inactive: list[int]
    ):
        """
        :param host_id:   ID of the executor handling this client
        :param client_id: Global client ID
        :param cpu_flops:       Static compute capacity (e.g. FLOPS baseline)
        :param gpu_flops:       Static GPU capacity (if used)
        :param timestamps_livelab: sorted timestamps (s) for network-rate changes
        :param rate:      upload/download rate trace (Mb/s) at each timestamp
        :param timestamps_carat:    sorted timestamps (s) for compute changes
        :param availability: CPU/GPU availability percentage (0–100) per timestamp
        :param batteryLevel:  battery percentage (0–100) per timestamp
        """
        # Identity
        self.host_id = host_id
        self.client_id = client_id
        self.size = size

        # Static capacities
        self.cpu_flops = cpu_flops
        self.gpu_flops = gpu_flops

        # Network traces (livelab)
        self.timestamps_livelab = timestamps_livelab
        self.rate = rate

        # Compute traces (carat)
        self.timestamps_carat = timestamps_carat
        self.availability = availability
        self.batteryLevel = batteryLevel

        # activity intervals
        self.active = active
        self.inactive = inactive

        # For adaptive sampling (e.g. Oort)
        self.score = 0

        # Noise to perturb FLOPS in one round
        self._round_noise = 1.0

        # most-recent end-to-end latency (sec)
        self.last_duration = None

    def get_score(self):
        return self.score

    def register_reward(self, reward: float):
        """Update the sampling score for this client."""
        self.score = reward

    def is_active(self, cur_time):
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
            [(ts, 'a') for ts in self.active] +
            [(ts, 'i') for ts in self.inactive]
        )

        # Initial phase
        phase = 'a' if (self.active and self.active[0] == 0) else 'i'

        # Walk through boundaries and flip the phase whenever we pass one
        for ts, _ in boundaries[1:]:  # skip the initial 0 entry
            if t < ts:
                break
            phase = 'i' if phase == 'a' else 'a'

        return phase == 'a'


    def _lookup(self, timestamps: list[int], values: list[float], t: float) -> float:
        """Return the trace value at sim-time t, assuming timestamps are sorted."""
        norm_t = t % (48*3600)
        if norm_t < timestamps[0]:
            return values[0]
        idx = max(i for i, ts in enumerate(timestamps) if ts <= norm_t)
        return values[idx]


    def bandwidth(self, t):
        rate = self._lookup(self.timestamps_livelab, self.rate, t)  # Mb/s
        return (rate / 8.0)  # MB/s

    def compute_speed(self, t: float) -> float:
        """
        Calculate effective FLOPS/s at time t, factoring in availability,
        battery level, and log-normal noise.
        """
        # 1) Base peak FLOPS
        base_peak = self.cpu_flops + self.gpu_flops  # CPU_FLOPS + GPU_FLOPS

        # 2) Availability fraction (0–1)
        avail_pct = self._lookup(self.timestamps_carat, self.availability, t) / 100.0

        # 3) Battery reduction factor
        batt = self._lookup(self.timestamps_carat, self.batteryLevel, t)
        if batt >= 70:
            batt_factor = 1.0
        elif batt >= 50:
            batt_factor = 0.9
        elif batt >= 30:
            batt_factor = 0.8
        elif batt >= 10:
            batt_factor = 0.6
        else:
            batt_factor = 0.4

        # 5) Final effective FLOPS/s
        return base_peak * avail_pct * batt_factor * self._round_noise


    def _simulate_data_phase(
        self,
        start_time: float,
        total_work: float,
        timestamps: list[int],
        rate_fn: Callable[[float], float],
        window: float,
        scale: float,
    ) -> float:
        """
        Generic simulator for download/upload or compute:
        - Loops over each interval where the rate_fn is constant,
        - Subtracts work done until total_work <= 0, then returns the exact finish time.

        Args:
            start_time: absolute sim time when phase begins.
            total_work: total MB (or total FLOPs) to complete.
            timestamps: breakpoints (in [0, window]) for when rate_fn may change.
            rate_fn: function t->rate (MB/s or FLOPS).
            window: cycle length (48h).
            scale: multiplier on rate_fn (e.g. 1.0).

        Returns:
            float: sim time when work_remaining hits zero.
        """
        # sort the cycle breakpoints
        pts = timestamps
        # normalize into window
        t0 = start_time % window
        abs_cycle_start = start_time - t0

        # find next index in pts after t0
        idx = next((i for i, x in enumerate(pts) if x > t0), len(pts))

        curr_time = start_time
        work_rem = total_work

        while True:
            # determine end of this sub-interval
            if idx < len(pts):
                next_point = abs_cycle_start + pts[idx]
            else:
                # wrap-around to end of window
                next_point = abs_cycle_start + window

            dt = next_point - curr_time
            rate = rate_fn(curr_time) * scale
            if rate <= 0:
                raise RuntimeError(f"Zero rate at t={curr_time}")

            potential = rate * dt
            if potential >= work_rem:
                # finishes within this interval
                return curr_time + (work_rem / rate)

            # subtract what we can do in this slice
            work_rem -= potential
            # advance time
            curr_time = next_point

            # if we wrapped around, shift the cycle
            if idx >= len(pts):
                abs_cycle_start += window
                t0 = 0
                idx = 0
            else:
                idx += 1

    def get_completion_time(
        self,
        cur_time: int,
        batch_size: int,
        local_steps: int,
        model_size: int,
        augmentation_factor: float = 3.0,
        reduction_factor: float = 0.5
    ) -> float:
        """
        Simulate download → local training → upload, using dynamic bandwidth and compute traces.

        Args:
            cur_time: simulation time (s) when download starts.
            batch_size: local batch size.
            local_steps: number of local training iterations.
            model_size: number of model parameters.
            augmentation_factor: multiplies forward-flop cost to include backward.
            reduction_factor: upload speed is bandwidth * this factor.

        Returns:
            float: simulation time (s) when upload finishes.
        """
        
        # Constants
        WINDOW = 48 * 3600  # 48h in seconds

        # Noise
        self._round_noise = np.random.lognormal(mean=0.0, sigma=0.25)

        # 1) Convert model size to MB (float16 = 2 bytes/param)
        total_bytes = model_size * 2
        model_mb = total_bytes / 1e6

        # 2) DOWNLOAD phase
        download_end = self._simulate_data_phase(
            start_time=cur_time,
            total_work=model_mb,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0  # download at full bandwidth
        )

        # 3) COMPUTE phase
        # total FLOPs = augmentation_factor × model_size × batch_size × local_steps
        total_ops = augmentation_factor * model_size * batch_size * local_steps
        compute_end = self._simulate_data_phase(
            start_time=download_end,
            total_work=total_ops,
            timestamps=self.timestamps_carat,
            rate_fn=self.compute_speed,
            window=WINDOW,
            scale=1.0  # compute_speed already in FLOPS
        )

        # 4) UPLOAD phase
        upload_end = self._simulate_data_phase(
            start_time=compute_end,
            total_work=model_mb * reduction_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0  # upload uses same units MB/s, reduction applied via total_work
        )

        return (upload_end-cur_time)

    def get_completion_time_lognormal(
        self,
        cur_time: float,
        batch_size: int,
        local_steps: int,
        model_size: int,
        augmentation_factor: float = 3.0,
        reduction_factor: float = 0.5,
        mean_seconds_per_sample: float = 0.005,
        tail_skew: float = 0.6,
    ) -> float:
        """
        Simulate download → lognormal‐based compute → upload.

        Compute time is sampled as:
          device_speed ~ LogNormal(mean=1, sigma=tail_skew)
          comp_time = device_speed
                    * mean_seconds_per_sample
                    * batch_size
                    * local_steps

        Returns the simulated time when upload completes.
        """

        # 1) Convert model_size to MB (float16 = 2 bytes/param)
        total_bytes = model_size * 2
        model_mb = total_bytes / 1e6

        # 2) DOWNLOAD phase (same as before)
        download_end = self._simulate_data_phase(
            start_time=cur_time,
            total_work=model_mb,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=48 * 3600,
            scale=1.0,
        )

        # 3) COMPUTE phase (lognormal sample)
        device_speed = max(0.0001, np.random.lognormal(mean=1.0, sigma=tail_skew))
        comp_time = device_speed * mean_seconds_per_sample * batch_size * local_steps
        compute_end = download_end + comp_time

        # 4) UPLOAD phase (same as before, with reduction)
        upload_end = self._simulate_data_phase(
            start_time=compute_end,
            total_work=model_mb * reduction_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=48 * 3600,
            scale=1.0,
        )

        return upload_end