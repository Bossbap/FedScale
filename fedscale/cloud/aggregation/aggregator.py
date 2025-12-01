# -*- coding: utf-8 -*-
import collections
import copy
import json
import os
import pickle
import random
import threading
import time
from concurrent import futures
import logging
import types

# Avoid noisy gRPC fork warnings when background threads are active.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_PYTHON_FORK_SUPPORT_ENABLED", "0")

import grpc
import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.aggregator_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.channels import job_api_pb2
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.fllibs import *

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    # ----------------------------------------------------------------------
    #  Construction / init
    # ----------------------------------------------------------------------
    def __init__(self, args):
        # init aggregator logger
        logger.initiate_aggregator_setting()

        resume_dir = getattr(args, "resume_from", "") or ""
        self._resume_state = None
        self._resume_plan = None
        self._resume_model_weights = None
        self._resume_optimizer_state = None
        self._resume_random_state = None
        self._resume_numpy_state = None
        self._resume_torch_state = None
        self._resume_cuda_states = None
        self._resume_client_manager_state = None
        self._resume_aggregator_state = None

        if resume_dir:
            resume_dir = os.path.abspath(resume_dir)
            self._resume_state = self._load_checkpoint(resume_dir)
            saved_args = copy.deepcopy(self._resume_state.get("args", {}))
            args = self._merge_args(args, saved_args)
            self._resume_plan = copy.deepcopy(self._resume_state.get("plan"))
            self._resume_model_weights = self._resume_state.get("model_weights")
            self._resume_optimizer_state = self._resume_state.get("optimizer_state")
            rng_pack = self._resume_state.get("rng", {})
            self._resume_random_state = rng_pack.get("python")
            self._resume_numpy_state = rng_pack.get("numpy")
            self._resume_torch_state = rng_pack.get("torch")
            self._resume_cuda_states = rng_pack.get("torch_cuda")
            self._resume_client_manager_state = self._resume_state.get("client_manager")
            self._resume_aggregator_state = self._resume_state.get("aggregator", {})
            self._checkpoint_dir = resume_dir
        else:
            self._checkpoint_dir = None

        logging.info(f"Job args {args}")
        self.args = args
        self.base_seed = getattr(self.args, "sample_seed", None)
        self.setup_seed(seed=self.base_seed)
        self.experiment_mode = self.args.experiment_mode
        self.device = self.args.cuda_device if self.args.use_cuda else torch.device("cpu")

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.0
        self.round_duration = 0.0
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=self.args)

        if self._resume_client_manager_state:
            self.client_manager.load_state(self._resume_client_manager_state)
            # keep aggregator args aligned with selector pacing state
            if self.client_manager.mode in ("oort", "pyramidfl"):
                sampler = getattr(
                    self.client_manager,
                    "pyr_sampler" if self.client_manager.mode == "pyramidfl" else "ucb_sampler",
                    None,
                )
                if sampler is not None and hasattr(sampler, "t_budget"):
                    self.args.t_budget = float(sampler.t_budget)
            elif self.client_manager.mode == "bliss":
                sampler = getattr(self.client_manager, "bliss_sampler", None)
                if sampler is not None and hasattr(sampler, "t_budget"):
                    self.args.t_budget = float(sampler.t_budget)

        if not self._checkpoint_dir:
            base_log_path = os.path.abspath(self.args.log_path)
            self._checkpoint_dir = os.path.join(
                base_log_path,
                "logs",
                self.args.job_name,
                self.args.time_stamp,
                "checkpoints",
            )
        self._checkpoint_file = os.path.join(self._checkpoint_dir, "latest.pt")
        self.checkpoint_interval = int(getattr(self.args, "checkpoint_interval", -1))

        # ======== model and data ========
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        self.model_weights = None
        self._aggregation_weight_sum = 0.0

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None
        # Map executor_id to peer IP observed at registration
        self.executor_peers = {}

        # ======== Event Queues =======
        self.individual_client_events = {}  # Unicast
        self.server_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # ======== runtime information ========
        self.tasks_round = 0
        self.num_of_clients = 0

        self.sampled_participants = []
        self.sampled_executors = []
        # ensure attribute exists for non-adaptive summary logging paths
        self.virtual_client_clock = {}
        self.flatten_client_duration = np.array([])

        self.round_stragglers = []
        self.model_size = 0
        self.model_amount_parameters = 0

        self.collate_fn = None
        self.round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # Adaptive‑training state
        self.pending_client_results = []

        # PyramidFL per-client (t_comp, t_total) cache for the current round
        self._pyramid_times = {}

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = []
        self.testing_history = {
            "data_set": args.data_set,
            "model": args.model,
            "sample_mode": args.sample_mode,
            "gradient_policy": args.gradient_policy,
            "task": args.task,
            "perf": collections.OrderedDict(),
        }

        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"aggregator{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
                self.wandb.define_metric("Agg/*",   step_metric="round")
                self.wandb.define_metric("AggWC/*", step_metric="clock")
                self.wandb.define_metric("round", hidden=True)
                self.wandb.define_metric("clock", hidden=True)

                self.wandb.config.update(
                    {
                        "num_participants": args.num_participants,
                        "data_set": args.data_set,
                        "model": args.model,
                        "gradient_policy": args.gradient_policy,
                        "eval_interval": args.eval_interval,
                        "rounds": args.rounds,
                        "batch_size": args.batch_size,
                        "use_cuda": args.use_cuda,
                    }
                )
            else:
                logging.error("Warning: wandb has already been initialized")
        else:
            self.wandb = None

        self.param_order = None

        # Apply resume-specific core state (round counters, learning rate, clocks)
        self._apply_resume_core_state()

        # ======== Task specific ============
        self.init_task_context()

        # ======== logging buffers (per-round) ============
        self._round_begin_payload = {}
        self._pre_util_map = {}              # {cid: util_before} for oort/pyramidfl
        self._client_round_utils = {}        # {cid: util_after}
        self._client_time_breakdown = {}     # {cid: {t_dl,t_comp,t_ul,(iters|iters_p1|iters_p2|dropout_frac)}}
        self._sampled_durations = {}
        self.completed_clients = []          # [cid]
        self.failed_clients = []             # [cid]
        self._bliss_pred_seen = {}           # {cid: pred_util}
        self._bliss_pred_unseen = {}         # {cid: pred_util}


    # ----------------------------------------------------------------------
    #  Helpers: structured logging utilities
    # ----------------------------------------------------------------------

    # ───────────────────────────────────────────────────────────────────
    # JSON log helper (stable, plot-friendly one-liners)
    # ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _json_default(o):
        import numpy as _np
        if isinstance(o, (set,)):
            return list(o)
        if isinstance(o, (_np.generic,)):
            return o.item()
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
        return str(o)

    def _sanitize_for_json(self, obj):
        import numpy as _np
        # dict: sanitize keys and values
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # fix key type
                if isinstance(k, _np.generic):
                    k = k.item()
                elif isinstance(k, bytes):
                    k = k.decode("utf-8", "ignore")
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    try:
                        # try numeric cast first
                        k = int(k)  # will work for numpy integers
                    except Exception:
                        k = str(k)
                out[k] = self._sanitize_for_json(v)
            return out
        # lists/tuples/sets → list of sanitized
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(x) for x in obj]
        # numpy arrays → list
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        # numpy scalars → python scalars
        if isinstance(obj, _np.generic):
            return obj.item()
        # passthrough builtin scalars / other objects
        return obj

    def _log_json(self, tag: str, payload: dict) -> None:
        try:
            safe = self._sanitize_for_json(payload)
            logging.info("[%s] %s", tag, json.dumps(safe, ensure_ascii=False))
        except Exception:
            logging.exception("[logging] failed to json-dump payload for tag=%s", tag)

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------
    def _merge_args(self, live_args, saved_args):
        """Merge runtime args with checkpoint args, keeping runtime-specific overrides.

        When resuming from a checkpoint, most arguments should come from the
        original run to ensure determinism and continuity. However, certain
        operational knobs must remain overrideable from the new invocation.
        In addition to network/runtime fields (e.g., ps_ip, executor layout),
        allow `rounds` to be increased/decreased so a run can continue past
        the originally planned number of rounds.
        """
        live_dict = copy.deepcopy(vars(live_args))
        runtime_overrides = {
            "ps_ip",
            "ps_port",
            "executor_configs",
            "this_rank",
            "num_executors",
            "cuda_device",
            "job_name",
            "time_stamp",
            "resume_from",
            "checkpoint_interval",
            # Allow changing total number of rounds upon resume
            "rounds",
        }
        for key, value in (saved_args or {}).items():
            if key in runtime_overrides:
                continue
            live_dict[key] = value
        return types.SimpleNamespace(**live_dict)

    def _load_checkpoint(self, directory: str):
        """Load and return the raw checkpoint payload from *directory*."""
        ckpt_file = os.path.join(directory, "latest.pt")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_file}. Ensure --resume_from points to a valid directory."
            )
        logging.info("Loading checkpoint from %s", ckpt_file)
        payload = torch.load(ckpt_file, map_location="cpu")
        version = payload.get("version", 0)
        if version != 1:
            raise ValueError(f"Unsupported checkpoint version {version} (expected 1)")
        return payload

    def _apply_resume_core_state(self):
        """Restore core counters, clocks and LR when resuming."""
        if not self._resume_state:
            return

        agg = self._resume_aggregator_state or {}
        try:
            self.round = int(agg.get("round", self.round))
        except Exception:
            pass
        try:
            self.global_virtual_clock = float(agg.get("global_virtual_clock", self.global_virtual_clock))
        except Exception:
            pass
        try:
            self.round_duration = float(agg.get("round_duration", self.round_duration))
        except Exception:
            pass
        try:
            self.tasks_round = int(agg.get("tasks_round", self.tasks_round))
        except Exception:
            pass

        lr = agg.get("learning_rate", None)
        if lr is not None:
            try:
                self.args.learning_rate = float(lr)
            except Exception:
                logging.warning("Failed to cast checkpoint learning rate %s", lr)

        logging.info(
            "Resuming from checkpoint: round=%d, clock=%s, lr=%s",
            self.round,
            self.global_virtual_clock,
            self.args.learning_rate,
        )
        self._resume_state = None

    def _collect_current_plan(self, clients_to_run, round_stragglers):
        """Capture the current round plan (after selection) for checkpointing."""
        plan = {
            "round": int(self.round),
            "mode": self.client_manager.mode,
            "adaptive_training": bool(self.args.adaptive_training),
            "sampled_participants": list(self.sampled_participants),
            "clients_to_run": list(clients_to_run),
            "round_stragglers": list(round_stragglers),
            "tasks_round": int(self.tasks_round),
            "virtual_client_clock": {int(k): float(v) for k, v in (self.virtual_client_clock or {}).items()},
            "round_duration": float(self.round_duration),
            "flatten_client_duration": self.flatten_client_duration.tolist() if isinstance(self.flatten_client_duration, np.ndarray) else list(self.flatten_client_duration or []),
            "client_time_breakdown": {int(k): copy.deepcopy(v) for k, v in (self._client_time_breakdown or {}).items()},
            "pre_util_map": {int(k): float(v) for k, v in (self._pre_util_map or {}).items()},
            "pyramid_times": {int(k): tuple(v) for k, v in (self._pyramid_times or {}).items()},
            "bliss_pred_seen": {int(k): float(v) for k, v in (self._bliss_pred_seen or {}).items()},
            "bliss_pred_unseen": {int(k): float(v) for k, v in (self._bliss_pred_unseen or {}).items()},
            "round_begin_payload": copy.deepcopy(self._round_begin_payload),
        }
        # include per-sampled durations for Oort/PyramidFL to restore selector state precisely
        if hasattr(self, "_sampled_durations") and self._sampled_durations:
            plan["sampled_durations"] = copy.deepcopy(self._sampled_durations)
        return plan

    def _apply_round_plan(self, plan: dict):
        """Apply a stored round plan (used when resuming)."""
        if not plan:
            raise ValueError("Empty round plan supplied for resume.")

        self.sampled_participants = list(plan.get("sampled_participants", []))
        self.tasks_round = int(plan.get("tasks_round", self.args.num_participants))
        self.round_duration = float(plan.get("round_duration", 0.0))
        self.round_stragglers = list(plan.get("round_stragglers", []))
        vc = plan.get("virtual_client_clock", {}) or {}
        self.virtual_client_clock = {int(k): float(v) for k, v in vc.items()}
        flatten = plan.get("flatten_client_duration", [])
        self.flatten_client_duration = np.array(flatten, dtype=float) if flatten else np.array([])
        ctb = plan.get("client_time_breakdown", {}) or {}
        self._client_time_breakdown = {int(k): copy.deepcopy(v) for k, v in ctb.items()}
        self._pre_util_map = {int(k): float(v) for k, v in (plan.get("pre_util_map", {}) or {}).items()}
        self._pyramid_times = {int(k): tuple(v) for k, v in (plan.get("pyramid_times", {}) or {}).items()}
        self._bliss_pred_seen = {int(k): float(v) for k, v in (plan.get("bliss_pred_seen", {}) or {}).items()}
        self._bliss_pred_unseen = {int(k): float(v) for k, v in (plan.get("bliss_pred_unseen", {}) or {}).items()}
        self._round_begin_payload = copy.deepcopy(plan.get("round_begin_payload", {}))

        # Reset per-round buffers
        self.completed_clients = []
        self.failed_clients = []
        self._client_round_utils = {}
        self.pending_client_results = []

        # Emit round-begin log to maintain continuity.
        if self._round_begin_payload:
            self._log_json("ROUND_BEGIN", self._round_begin_payload)

        clients_to_run = list(plan.get("clients_to_run", []))

        # Ensure selector sees the same per-client durations as in continuous run
        try:
            if self.client_manager.mode in ("oort", "pyramidfl"):
                # First: durations as used for top-K/stragglers
                if self.virtual_client_clock:
                    for cid, dur in self.virtual_client_clock.items():
                        self.client_manager.registerDuration(int(cid), float(dur))
                # Then: durations for all sampled clients captured during planning
                sd = plan.get("sampled_durations", {}) or {}
                if sd:
                    for cid_s, dur in sd.items():
                        self.client_manager.registerDuration(int(cid_s), float(dur))
                    # Log head of restored durations
                    head = sorted([(int(k), float(v)) for k, v in sd.items()])[:20]
                    logging.info("[RESUME_DURATIONS_HEAD] size=%d head=%s", len(sd), head)
        except Exception:
            logging.exception("[resume] failed to register per-client durations into selector")
        # Defer task assignment to dispatch time (we need executor set)
        return clients_to_run

    def _checkpoint_payload(self, plan: dict) -> dict:
        """Build the payload that will be persisted to disk."""
        args_snapshot = copy.deepcopy(vars(self.args))
        model_weights = []
        for w in self.model_wrapper.get_weights():
            if isinstance(w, torch.Tensor):
                model_weights.append(w.detach().cpu().numpy())
            else:
                model_weights.append(np.asarray(w, dtype=np.float32))

        rng_pack = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            try:
                rng_pack["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                logging.warning("Failed to capture CUDA RNG state for checkpoint.")

        payload = {
            "version": 1,
            "saved_at": time.time(),
            "args": args_snapshot,
            "rng": rng_pack,
            "aggregator": {
                "round": int(self.round),
                "global_virtual_clock": float(self.global_virtual_clock),
                "round_duration": float(self.round_duration),
                "tasks_round": int(self.tasks_round),
                "learning_rate": float(self.args.learning_rate),
            },
            "plan": copy.deepcopy(plan),
            "client_manager": self.client_manager.get_state(),
            "model_weights": model_weights,
            "optimizer_state": self.model_wrapper.get_optimizer_state(),
        }
        return payload

    def _save_checkpoint(self, plan: dict):
        """Persist the current training state to disk."""
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        except Exception:
            logging.exception("Failed to create checkpoint directory %s", self._checkpoint_dir)
            return

        payload = self._checkpoint_payload(plan)
        tmp_path = os.path.join(
            self._checkpoint_dir,
            f".tmp_ckpt_{int(time.time() * 1000)}.pt",
        )
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self._checkpoint_file)
            logging.info("Checkpoint saved for round %d at %s", self.round, self._checkpoint_file)
        except Exception:
            logging.exception("Failed to save checkpoint to %s", self._checkpoint_file)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _maybe_checkpoint(self, plan: dict):
        """Checkpoint at configured intervals."""
        if self.checkpoint_interval is None or self.checkpoint_interval <= 0:
            return
        interval = max(1, int(self.checkpoint_interval))
        if self.round % interval != 0:
            return
        self._save_checkpoint(plan)


    # ----------------------------------------------------------------------
    #  Basic env / comms
    # ----------------------------------------------------------------------
    def setup_env(self):
        """Set up experiments environment and server optimizer"""
        self.setup_seed(seed=self.base_seed)
        if self._resume_random_state is not None:
            try:
                random.setstate(self._resume_random_state)
            except Exception:
                logging.exception("Failed to restore Python RNG state from checkpoint")
            self._resume_random_state = None
        if self._resume_numpy_state is not None:
            try:
                np.random.set_state(self._resume_numpy_state)
            except Exception:
                logging.exception("Failed to restore NumPy RNG state from checkpoint")
            self._resume_numpy_state = None
        if self._resume_torch_state is not None:
            try:
                torch.random.set_rng_state(self._resume_torch_state)
            except Exception:
                logging.exception("Failed to restore Torch RNG state from checkpoint")
            self._resume_torch_state = None
        if self._resume_cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.random.set_rng_state_all(self._resume_cuda_states)
            except Exception:
                logging.warning("Unable to restore CUDA RNG states during resume.")
            self._resume_cuda_states = None

    def setup_seed(self, seed=None):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        if seed is None:
            seed = 233
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        logging.info("Initiating control plane communication ...")
        if self.experiment_mode == commons.SIMULATION_MODE:
            num_of_executors = 0
            for ip_numgpu in self.args.executor_configs.split("="):
                ip, numgpu = ip_numgpu.split(":")
                for numexe in numgpu.strip()[1:-1].split(","):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args.num_participants))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = "[::]:{}".format(self.args.ps_port)

        logging.info("%%%%%%%%%% Opening aggregator server using port %s %%%%%%%%%%", port)

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()

    def init_data_communication(self):
        """For jumbo traffics (e.g., training results)."""
        pass

    def init_model(self):
        """Initialize the model and **possibly** restore from checkpoint."""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = TensorflowModelAdapter(init_model())
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device
                ),
            )
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")

        self.model_weights = self.model_wrapper.get_weights()

        if self._resume_model_weights is not None:
            try:
                # Restore weights verbatim; avoid applying server optimizer here
                self.model_wrapper.set_weights(self._resume_model_weights, is_aggregator=False)
                self.model_weights = self.model_wrapper.get_weights()
                logging.info("Restored model weights from checkpoint.")
            except Exception:
                logging.exception("Failed to restore model weights from checkpoint.")
            self._resume_model_weights = None

        if self._resume_optimizer_state:
            try:
                self.model_wrapper.load_optimizer_state(self._resume_optimizer_state)
                logging.info("Restored optimizer state from checkpoint.")
            except Exception:
                logging.exception("Failed to restore optimizer state from checkpoint.")
            self._resume_optimizer_state = None

        if self.args.engine == commons.TENSORFLOW:
            self.model_amount_parameters = sum(w.size for w in self.model_weights)
            self.model_size = int(sum(w.size * w.dtype.size for w in self.model_weights) * 8 / 1_000_000)  # Mb
        elif self.args.engine == commons.PYTORCH:
            self.model_amount_parameters = sum(w.numel() for w in self.model_weights)
            self.model_size = int(sum(w.numel() * w.element_size() for w in self.model_weights)* 8 / 1_000_000)  # Mb
        logging.info("model amount of parameters: %d, model size: %f",
                     self.model_amount_parameters, self.model_size)

    def init_task_context(self):
        """Initiate execution context for specific tasks"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ["DATA_DIR", self.args.data_dir], server=True
            )

    def init_client_manager(self, args):
        """Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            ClientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """

        # sample_mode: random, oort, pyramidfl or bliss

        return ClientManager(
            args.sample_mode,
            args=args,
            sample_seed=getattr(args, "sample_seed", 233),
        )

    # ======================================================================
    #                     A D A P T I V E   H E L P E R S
    # ======================================================================
    def _collect_adaptive_result(self, res):
        """Buffer every upload; when all replies are in, aggregate."""
        cid = res["client_id"]
        if res.get("wall_duration", 0) > 0:
            self.client_manager.registerDuration(cid, res["wall_duration"])
        self.pending_client_results.append(res)

        # Wait until *every* sampled participant has replied
        if len(self.pending_client_results) < len(self.sampled_participants):
            return
        self._finalise_adaptive_round()

    def _record_non_adaptive_time_breakdown(self, clients_to_run, round_stragglers):
        """Copy the most recent simulated download/compute/upload breakdown for logging."""
        self._client_time_breakdown = {}
        candidate_ids = list(clients_to_run) + list(round_stragglers)
        seen = set()

        for cid in candidate_ids:
            cid_int = int(cid)
            if cid_int in seen:
                continue
            seen.add(cid_int)

            try:
                meta = self.client_manager.getClient(cid_int)
            except KeyError:
                continue

            getter = getattr(meta, "get_last_time_breakdown", None)
            breakdown = getter() if callable(getter) else None
            if not breakdown:
                continue

            t_dl = float(breakdown.get("t_dl", 0.0))
            t_comp = float(breakdown.get("t_comp", 0.0))
            t_ul = float(breakdown.get("t_ul", 0.0))
            iters = int(breakdown.get("local_steps", getattr(self.args, "local_steps", 0)))
            dropf = float(breakdown.get("dropout_frac", 0.0))

            if self.client_manager.mode != "pyramidfl":
                dropf = 0.0

            self._client_time_breakdown[cid_int] = {
                "t_dl": t_dl,
                "t_comp": t_comp,
                "t_ul": t_ul,
                "iters": iters,
                "dropout_frac": dropf,
            }

    def _finalise_adaptive_round(self):
        """Filter to the K fastest active clients, aggregate them, feed back stragglers."""
        K = self.args.num_participants

        pool = [r for r in self.pending_client_results if r.get("success", True)]

        active_pool = []
        for r in pool:
            finish_t = self.global_virtual_clock + r["wall_duration"]
            if self.client_manager.isClientActive(r["client_id"], finish_t):
                active_pool.append(r)

        kept = sorted(active_pool, key=lambda x: x["wall_duration"])[:K]
        stragglers = [r for r in self.pending_client_results if r not in kept]

        # group ids
        self.completed_clients = [int(r["client_id"]) for r in kept]
        self.failed_clients    = [int(r["client_id"]) for r in self.pending_client_results
                                  if not r.get("success", True)]
        self.round_stragglers  = [int(r["client_id"]) for r in stragglers
                                  if r.get("success", True)]

        self.tasks_round = len(kept)
        self.round_duration = max(r["wall_duration"] for r in kept) if kept else 0.0
        self.flatten_client_duration = np.array([r["wall_duration"] for r in kept])
        self.virtual_client_clock = {
            int(r["client_id"]): r["wall_duration"] for r in self.pending_client_results
        }

        self.model_in_update = 0
        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        avg_util = (sum(r["utility"] for r in kept) / max(1, len(kept))) if kept else 0.0

        # collect per-client utilities  time breakdown for logging
        self._client_round_utils = {}
        self._client_time_breakdown = {}
        for r in self.pending_client_results:
            cid = int(r["client_id"])
            self._client_round_utils[cid] = float(r.get("utility", 0.0))
            # executor returns detailed times & iterations for adaptive path
            self._client_time_breakdown[cid] = {
                "t_dl": float(r.get("t_dl", 0.0)),
                "t_comp": float(r.get("t_comp", 0.0)),
                "t_ul": float(r.get("t_ul", 0.0)),
                # phase info is Bliss-only but harmless to include
                "iters_p1": int(r.get("iters_phase1", 0)),
                "iters_p2": int(r.get("iters_phase2", 0)),
                "dropout_frac": float(r.get("dropout_frac", 0.0)),
            }

        for r in kept:
            cid = int(r["client_id"])

            if self.args.gradient_policy in ["q-fedavg"]:
                self.client_training_results.append(r)

            self.stats_util_accumulator.append(r["utility"])
            self.loss_accumulator.append(r["moving_loss"])

            self.client_manager.register_feedback(
                cid,
                r["utility"],
                time_stamp=self.round,
                duration=r["wall_duration"],
                success=True,
            )

            with self.update_lock:
                self.model_in_update += 1
                self.update_weight_aggregation(r)

        for r in stragglers:
            self.client_manager.register_feedback(
                int(r["client_id"]),
                avg_util,
                time_stamp=self.round,
                duration=r["wall_duration"],
                success=False,
            )

        self.pending_client_results.clear()

    # ----------------------------------------------------------------------
    #  Registration / executor info
    # ----------------------------------------------------------------------
    def client_register_handler(self, executorId, info):
        """Triggered once we receive new executor registration."""
        logging.info(f"Loading {len(info['size'])} client traces ...")

        # Use the actual ids provided by executors
        for real_id, _size in zip(info["client_ids"], info["size"]):
            self.client_manager.register_client(
                host_id=executorId, client_id=real_id, size=_size
            )
            self.num_of_clients += 1

        logging.info("Info of all feasible clients %s", self.client_manager.getDataInfo())

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executorId)
        ip = self.executor_peers.get(executorId, "unknown")
        logging.info(
            "Received executor %s information (ip=%s), %d/%d",
            executorId,
            ip,
            len(self.registered_executor_info),
            len(self.executors),
        )

        if self.experiment_mode == commons.SIMULATION_MODE:
            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                self.round_completion_handler()
        else:
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()

    # ----------------------------------------------------------------------
    #  Round time / stragglers estimation (SIMULATION_MODE)
    # ----------------------------------------------------------------------
    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):

        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).

        """


        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        sampled_durations = {}

        for client_to_run in sampled_clients:
            # Default args
            batch_size = self.args.batch_size
            local_steps = self.args.local_steps
            dropout_p = 0.0

            # PyramidFL per-client overrides (if any)
            if self.client_manager.mode == "pyramidfl":
                ov = self.client_manager.get_pyramidfl_conf(client_to_run)
                if ov:
                    local_steps = int(ov.get("local_steps", local_steps))
                    dropout_p = float(ov.get("dropout_p", 0.0))

                # Simulate completion time
                t_comp, roundDuration = self.client_manager.get_times_pyramid(
                    client_id=client_to_run,
                    cur_time=self.global_virtual_clock,
                    batch_size=batch_size,
                    local_steps=local_steps,
                    model_size=self.model_size,
                    model_amount_parameters=self.model_amount_parameters,
                    dropout_p=dropout_p,
                )
                # cache for feedback
                self._pyramid_times[client_to_run] = (t_comp, roundDuration)

            else:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                roundDuration = self.client_manager.get_completion_time(
                    client_to_run,
                    cur_time=self.global_virtual_clock,
                    batch_size=getattr(client_cfg, "batch_size", batch_size),
                    local_steps=getattr(client_cfg, "local_steps", local_steps),
                    model_size=self.model_size,
                    model_amount_parameters=self.model_amount_parameters,
                )

            if self.client_manager.mode in ("oort", "pyramidfl"):
                self.client_manager.registerDuration(client_to_run, duration=roundDuration)
            # record duration for all sampled clients (active or not)
            sampled_durations[int(client_to_run)] = float(roundDuration)

            if self.client_manager.isClientActiveThroughout(
                client_to_run,
                start_time=self.global_virtual_clock,
                end_time=self.global_virtual_clock + roundDuration,
            ):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[int(client_to_run)] = roundDuration

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
        workers_sorted_by_completion_time = sorted(
            range(len(completionTimes)), key=lambda k: completionTimes[k]
        )
        top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]
        stragglers = [
            sampledClientsReal[k]
            for k in workers_sorted_by_completion_time[num_clients_to_collect:]
        ]
        round_duration = completionTimes[top_k_index[-1]] if top_k_index else 0.0

        return (
            clients_to_run,
            stragglers,
            completed_client_clock,
            round_duration,
            [completionTimes[k] for k in top_k_index],
            sampled_durations,
        )

    # ----------------------------------------------------------------------
    #  Main loop
    # ----------------------------------------------------------------------
    def run(self):
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()
        self.init_model()

        self.event_monitor()
        self.stop()

    # ----------------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------------
    def _is_first_result_in_round(self):
        return self.model_in_update == 1

    def _is_last_result_in_round(self):
        return self.model_in_update == self.tasks_round

    def select_participants(self, select_num_participants, overcommitment=1.3):
        return sorted(
            self.client_manager.select_participants(
                int(select_num_participants * overcommitment),
                cur_time=self.global_virtual_clock,
            ),
        )

    # ----------------------------------------------------------------------
    #  Client completion / aggregation
    # ----------------------------------------------------------------------
    def client_completion_handler(self, results):
        # adaptive: buffer and handle later
        if self.args.adaptive_training:
            self._collect_adaptive_result(results)
            return

        if self.args.gradient_policy in ["q-fedavg"]:
            self.client_training_results.append(results)

        self.stats_util_accumulator.append(results["utility"])
        self.loss_accumulator.append(results["moving_loss"])
        cid = int(results["client_id"])

        dur_for_feedback = self.virtual_client_clock[cid]

        # debug logging removed

        # record for summary
        self._client_round_utils[cid] = float(results.get("utility", 0.0))
        if results.get("success", True):
            self.completed_clients.append(cid)
        else:
            self.failed_clients.append(cid)

        extra_kwargs = {}
        if self.client_manager.mode == "pyramidfl":
            t_pair = self._pyramid_times.get(cid, None)
            if t_pair:
                extra_kwargs["t_comp"] = t_pair[0]
                extra_kwargs["t_total"] = t_pair[1]
            if "gsize" in results:
                extra_kwargs["gsize"] = results["gsize"]
            # Also forward actual iterations completed so PyramidFL can
            # estimate per-step compute time correctly for the next round
            if "iters" in results:
                try:
                    extra_kwargs["steps"] = int(results["iters"])  # alias for clarity in selector
                except Exception:
                    pass
            # Update per-round time breakdown with actual iterations if planned
            try:
                if cid in self._client_time_breakdown:
                    self._client_time_breakdown[cid]["iters"] = int(results.get("iters", self._client_time_breakdown[cid].get("iters", 0)))
            except Exception:
                pass

        self.client_manager.register_feedback(
            cid,
            results["utility"],
            time_stamp=self.round,
            duration=dur_for_feedback,
            success=bool(results.get("success", True)),
            **extra_kwargs,
        )

        with self.update_lock:
            self.model_in_update += 1
            self.update_weight_aggregation(results)

    def update_weight_aggregation(self, results):

        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """

        # debug logging removed

        upd = results["update_weight"]
        weight = float(results.get("trained_size", 0.0) or 0.0)
        if weight <= 0.0:
            fallback = getattr(self.args, "local_steps", 0) * getattr(
                self.args, "batch_size", 0
            )
            weight = float(fallback) if fallback > 0 else 1.0

        if self._is_first_result_in_round():
            self._aggregation_weight_sum = weight
            if isinstance(upd, dict):
                self.param_order = list(upd.keys())
                self.model_weights = {
                    k: (
                        (v.copy() if hasattr(v, "copy") else copy.deepcopy(v)) * weight
                    )
                    for k, v in upd.items()
                }
            else:
                self.param_order = list(range(len(upd)))
                self.model_weights = [
                    (w.copy() if hasattr(w, "copy") else copy.deepcopy(w)) * weight
                    for w in upd
                ]
            # debug logging removed
            return

        self._aggregation_weight_sum += weight
        if isinstance(upd, dict):
            for k, v in upd.items():
                self.model_weights[k] += v * weight
        else:
            for idx, v in enumerate(upd):
                self.model_weights[idx] += v * weight

        # debug logging removed

        if not self._is_last_result_in_round():
            return

        denom = (
            float(self._aggregation_weight_sum) if self._aggregation_weight_sum else 1.0
        )

        if isinstance(self.model_weights, dict):
            to_send = []
            for k in self.param_order:
                v = self.model_weights[k]
                if np.issubdtype(v.dtype, np.integer):
                    v = v.astype(np.float32)
                self.model_weights[k] = v / denom
                to_send.append(self.model_weights[k])
        else:
            to_send = [w / denom for w in self.model_weights]

        if self.args.gradient_policy in ["q-fedavg"]:
            self.client_training_results.sort(
                key=lambda r: int(r.get("client_id", 0))
            )

        # debug logging removed

        self.model_wrapper.set_weights(
            copy.deepcopy(to_send),
            client_training_results=self.client_training_results,
        )
        self._aggregation_weight_sum = 0.0

    # ----------------------------------------------------------------------
    #  Testing
    # ----------------------------------------------------------------------
    def aggregate_test_result(self):
        accumulator = self.test_result_accumulator[0]
        for i in range(1, len(self.test_result_accumulator)):
            if self.args.task == "detection":
                for key in accumulator:
                    if key == "boxes":
                        for j in range(596):
                            accumulator[key][j] = (
                                accumulator[key][j]
                                + self.test_result_accumulator[i][key][j]
                            )
                    else:
                        accumulator[key] += self.test_result_accumulator[i][key]
            else:
                for key in accumulator:
                    accumulator[key] += self.test_result_accumulator[i][key]

        self.testing_history["perf"][self.round] = {
            "round": self.round,
            "clock": self.global_virtual_clock,
        }
        for metric_name in accumulator.keys():
            if metric_name == "test_loss":
                self.testing_history["perf"][self.round]["loss"] = (
                    accumulator["test_loss"]
                    if self.args.task == "detection"
                    else accumulator["test_loss"] / accumulator["test_len"]
                )
            elif metric_name not in ["test_len"]:
                self.testing_history["perf"][self.round][metric_name] = (
                    accumulator[metric_name] / accumulator["test_len"]
                )

        round_perf = self.testing_history["perf"][self.round]
        logging.info(
            "FL Testing in round: %d, virtual_clock: %s, results: %s",
            self.round,
            self.global_virtual_clock,
            round_perf,
        )

    # ----------------------------------------------------------------------
    #  LR decay
    # ----------------------------------------------------------------------
    def update_default_task_config(self):
        """Apply LR decay etc. at the *beginning* of each round."""
        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor,
                self.args.min_learning_rate,
            )

    # ----------------------------------------------------------------------
    #  Round completion handler (unified)
    # ----------------------------------------------------------------------
    def round_completion_handler(self):
        """
        Two roles:

        (1) **Bootstrap / Resume at beginning-of-round**:
            - No updates collected (tasks_round == 0 and stats empty).
            - We DO NOT increment `round`, nor decay here (round is already the current one).
            - We **can** checkpoint if interval hits (round already points to current round).
            - We select & dispatch the same round.

        (2) **Normal completion**:
            - We have all updates; we advance the clock, increment `round`,
              apply LR decay for the new round, checkpoint, then select & dispatch it.
        """
        has_resume_plan = bool(self._resume_plan)
        bootstrap = has_resume_plan or (
            self.tasks_round == 0 and len(self.stats_util_accumulator) == 0
        )

        # ------------------------------------------------------------------
        # 0) BOOTSTRAP / RESUME
        # ------------------------------------------------------------------
        if bootstrap:
            logging.info("Bootstrap: BEGINNING of round %d", self.round)

            resumed_plan = False

            if has_resume_plan:
                plan = copy.deepcopy(self._resume_plan)
                self._resume_plan = None
                clients_to_run = self._apply_round_plan(plan)
                round_stragglers = list(plan.get("round_stragglers", []))
                resumed_plan = True
            else:
                online_clients = self.client_manager.getOnlineClients(self.global_virtual_clock)

                self.sampled_participants = self.select_participants(
                    select_num_participants=self.args.num_participants,
                    overcommitment=self.args.overcommitment,
                )

                round_stragglers = []
                if self.args.adaptive_training:
                    clients_to_run = self.sampled_participants
                    self.tasks_round = self.args.num_participants
                    self.virtual_client_clock = {}
                    self.flatten_client_duration = np.array([])
                    self.round_duration = 0.0
                    self.round_stragglers = []
                else:
                    (
                        clients_to_run,
                        round_stragglers,
                        virtual_client_clock,
                        round_duration,
                        flatten_client_duration,
                        sampled_durations,
                    ) = self.tictak_client_tasks(
                        self.sampled_participants, self.args.num_participants
                    )
                    self.tasks_round = len(clients_to_run)
                    self.virtual_client_clock = virtual_client_clock
                    self.flatten_client_duration = np.array(flatten_client_duration)
                    self.round_duration = round_duration
                    self.round_stragglers = round_stragglers
                    self._sampled_durations = sampled_durations

                    self._record_non_adaptive_time_breakdown(clients_to_run, round_stragglers)

                self._pre_util_map = {}
                if self.client_manager.mode in ("oort", "pyramidfl"):
                    try:
                        metrics = self.client_manager.getAllMetrics() or {}
                        # include both run and straggler candidates (active throughout)
                        for cid in list(clients_to_run) + list(round_stragglers):
                            rec = metrics.get(cid, {})
                            self._pre_util_map[int(cid)] = float(rec.get("reward", 0.0))
                    except Exception:
                        logging.exception("[logging] could not gather pre-utility map")

                # ── Bliss predictions (seen/unseen) for this round, when applicable
                self._bliss_pred_seen, self._bliss_pred_unseen = {}, {}
                if self.client_manager.mode == "bliss":
                    try:
                        m = self.client_manager.getAllMetrics() or {}
                        self._bliss_pred_seen = dict(m.get("pred_seen", {}))
                        self._bliss_pred_unseen = dict(m.get("pred_unseen", {}))
                    except Exception:
                        logging.exception("[logging] could not retrieve Bliss predictions")

                # Reset per-round accumulators that will be filled during the round
                self.completed_clients = []
                self.failed_clients = []
                self._client_round_utils = {}

                # Structured round-begin log
                self._round_begin_payload = {
                    "round": int(self.round),
                    "start_clock": float(self.global_virtual_clock),
                    "pacer": self.client_manager.get_pacer_state(),
                }
                self._log_json("ROUND_BEGIN", self._round_begin_payload)

            # Ensure selector sees identical sampled durations as in the original planning
            try:
                if has_resume_plan and self.client_manager.mode in ("oort", "pyramidfl"):
                    # durations for all sampled participants were captured in plan
                    sd = plan.get("sampled_durations", {}) or {}
                    for cid_s, dur in sd.items():
                        self.client_manager.registerDuration(int(cid_s), float(dur))
            except Exception:
                logging.exception("[resume] failed to register sampled durations into selector")

            self.resource_manager.register_tasks(clients_to_run)

            if self.experiment_mode == commons.SIMULATION_MODE:
                self.sampled_executors = list(self.individual_client_events.keys())
            else:
                self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

            self.model_in_update = 0
            self.test_result_accumulator = []
            self.stats_util_accumulator = []
            self.client_training_results = []
            self.loss_accumulator = []

            logging.info(
                "Selected %d participants to run: %s",
                len(clients_to_run),
                clients_to_run,
            )

            # Start round
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            if self.round % self.args.eval_interval == 0:
                self.broadcast_aggregator_events(commons.MODEL_TEST)
            else:
                self.broadcast_aggregator_events(commons.START_ROUND)
            return

        # ------------------------------------------------------------------
        # 1) NORMAL COMPLETION
        # ------------------------------------------------------------------

        # Advance wall-clock & bump round number
        self.global_virtual_clock += self.round_duration
        completed_round = int(self.round)
        self.round += 1

        # self.log_control_state("rc.on-entry")

        # Feedback to stragglers
        last_round_avg_util = sum(self.stats_util_accumulator) / max(
            1, len(self.stats_util_accumulator)
        )
        for client_id in self.round_stragglers:
            self.client_manager.register_feedback(
                client_id,
                last_round_avg_util,
                time_stamp=completed_round,
                duration=self.virtual_client_clock[client_id],
                success=False,
            )

        # Log loss / training stats
        avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))

        prev_round = completed_round
        clock_end = float(self.global_virtual_clock)
        duration = float(self.round_duration)
        clock_start = float(clock_end - duration)
        meta_payload = {
            "round": prev_round,
            "clock_start": clock_start,
            "clock_end": clock_end,
            "duration": duration,
            "loss_avg": float(avg_loss),
            "completed": sorted(set(self.completed_clients)),
            "stragglers": sorted(set(self.round_stragglers)),
            "dropped": sorted(set(self.failed_clients)),
        }

        # Emit per-client information for this round (mode-specific) with metadata
        try:
            mode = self.client_manager.mode
            info_key = f"{mode}_client_information"

            if self.args.adaptive_training:
                selected_ids = sorted({int(cid) for cid in self.sampled_participants})
            else:
                selected_ids = sorted({int(cid) for cid in self.virtual_client_clock.keys()})

            status_map = {}
            for sid in selected_ids:
                if sid in self.failed_clients:
                    status_map[sid] = 'd'
                elif sid in self.completed_clients:
                    status_map[sid] = 'c'
                elif sid in self.round_stragglers:
                    status_map[sid] = 's'
                else:
                    status_map[sid] = 's'

            pre_util = {int(k): float(v) for k, v in (self._pre_util_map or {}).items()}
            post_util = {}
            if mode in ("oort", "pyramidfl"):
                try:
                    metrics = self.client_manager.getAllMetrics() or {}
                except Exception:
                    logging.exception("[logging] could not gather post-utility map; falling back to raw client utilities")
                    metrics = {}
                for sid in selected_ids:
                    val = None
                    if metrics:
                        rec = metrics.get(sid, {}) or {}
                        val = rec.get("reward", None)
                    if val is None:
                        val = self._client_round_utils.get(sid, 0.0)
                    post_util[sid] = float(val)

            adaptive_training = bool(getattr(self.args, "adaptive_training", False))
            run_phase_2 = bool(getattr(self.args, "run_phase_2", False)) if adaptive_training else False
            client_info = {}
            for sid in selected_ids:
                tb = self._client_time_breakdown.get(sid, {}) or {}
                t_dl = float(tb.get("t_dl", 0.0))
                t_comp = float(tb.get("t_comp", 0.0))
                t_ul = float(tb.get("t_ul", 0.0))
                st = status_map.get(sid, "s")

                iters_phase1 = 0
                iters_phase2 = 0
                dropout_frac = 0.0

                if adaptive_training:
                    # Executors report per-phase iterations for adaptive runs.
                    iters_phase1 = int(tb.get("iters_phase1", tb.get("iters_p1", 0)))
                    if run_phase_2:
                        iters_phase2 = int(tb.get("iters_phase2", tb.get("iters_p2", 0)))
                        dropout_frac = float(tb.get("dropout_frac", 0.0))
                    else:
                        iters_phase2 = 0
                        dropout_frac = 0.0
                else:
                    if mode == "pyramidfl":
                        iters_phase1 = int(tb.get("iters", tb.get("local_steps", getattr(self.args, "local_steps", 0))))
                        iters_phase2 = 0
                        dropout_frac = float(tb.get("dropout_frac", 0.0))
                    else:
                        iters_phase1 = 0
                        iters_phase2 = 0
                        dropout_frac = 0.0

                predicted_utility = 0.0
                computed_utility = 0.0

                if mode == "random":
                    predicted_utility = 0.0
                    computed_utility = 0.0
                elif mode in ("oort", "pyramidfl"):
                    predicted_utility = float(pre_util.get(sid, 0.0))
                    computed_utility = float(post_util.get(sid, self._client_round_utils.get(sid, 0.0)))
                elif mode == "bliss":
                    if sid in (self._bliss_pred_seen or {}):
                        predicted_utility = float(self._bliss_pred_seen[sid])
                    elif sid in (self._bliss_pred_unseen or {}):
                        predicted_utility = float(self._bliss_pred_unseen[sid])
                    computed_utility = float(self._client_round_utils.get(sid, 0.0))
                else:
                    computed_utility = float(self._client_round_utils.get(sid, 0.0))

                client_info[sid] = [
                    t_dl,
                    t_comp,
                    t_ul,
                    iters_phase1,
                    iters_phase2,
                    dropout_frac,
                    predicted_utility,
                    computed_utility,
                    st,
                ]

            payload = {
                "round": meta_payload["round"],
                "clock_start": meta_payload["clock_start"],
                "clock_end": meta_payload["clock_end"],
                "duration": meta_payload["duration"],
                "loss_avg": meta_payload["loss_avg"],
                # "completed": meta_payload["completed"],
                # "stragglers": meta_payload["stragglers"],
                # "dropped": meta_payload["dropped"],
                info_key: client_info,
            }
            self._log_json("CLIENT_INFO", payload)
        except Exception:
            logging.exception("[logging] failed to emit per-client information payload")

        # reset per-round buffers (safety)
        self._pre_util_map.clear()
        self._client_round_utils.clear()
        self._client_time_breakdown.clear()
        self.completed_clients.clear()
        self.failed_clients.clear()
        self._bliss_pred_seen.clear()
        self._bliss_pred_unseen.clear()

        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # --- Apply LR decay (& any other beginning-of-round mutations) ---
        self.update_default_task_config()

        # ======= Now select and dispatch the *next* round =======
        self._pyramid_times = {}

        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants,
            overcommitment=self.args.overcommitment,
        )

        if self.args.adaptive_training:
            clients_to_run = self.sampled_participants
            self.tasks_round = self.args.num_participants
            self.virtual_client_clock = {}
            self.flatten_client_duration = np.array([])
            self.round_duration = 0.0
            self.round_stragglers = []
            self._client_time_breakdown = {}
        else:
            (
                clients_to_run,
                round_stragglers,
                virtual_client_clock,
                round_duration,
                flatten_client_duration,
                sampled_durations,
            ) = self.tictak_client_tasks(
                self.sampled_participants, self.args.num_participants
            )

            self.tasks_round = len(clients_to_run)
            self.virtual_client_clock = virtual_client_clock
            self.flatten_client_duration = np.array(flatten_client_duration)
            self.round_duration = round_duration
            self.round_stragglers = round_stragglers
            self._sampled_durations = sampled_durations
            self._record_non_adaptive_time_breakdown(clients_to_run, round_stragglers)

        self._pre_util_map = {}
        if self.client_manager.mode in ("oort", "pyramidfl"):
            try:
                metrics = self.client_manager.getAllMetrics() or {}
                for cid in list(clients_to_run) + list(self.round_stragglers):
                    rec = metrics.get(cid, {})
                    self._pre_util_map[int(cid)] = float(rec.get("reward", 0.0))
            except Exception:
                logging.exception("[logging] could not gather pre-utility map")

        self._bliss_pred_seen, self._bliss_pred_unseen = {}, {}
        if self.client_manager.mode == "bliss":
            try:
                m = self.client_manager.getAllMetrics() or {}
                self._bliss_pred_seen = dict(m.get("pred_seen", {}))
                self._bliss_pred_unseen = dict(m.get("pred_unseen", {}))
            except Exception:
                logging.exception("[logging] could not retrieve Bliss predictions")

        self.completed_clients = []
        self.failed_clients = []
        self._client_round_utils = {}

        _ = self.client_manager.getOnlineClients(self.global_virtual_clock)
        self._round_begin_payload = {
            "round": int(self.round),
            "start_clock": float(self.global_virtual_clock),
            "pacer": self.client_manager.get_pacer_state(),
        }
        self._log_json("ROUND_BEGIN", self._round_begin_payload)

        plan_to_save = self._collect_current_plan(clients_to_run, self.round_stragglers)
        self._maybe_checkpoint(plan_to_save)

        logging.info(
            "Selected %d participants to run: %s",
            len(clients_to_run),
            clients_to_run,
        )

        self.resource_manager.register_tasks(clients_to_run)

        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []

        # Stop or continue
        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    # ----------------------------------------------------------------------
    #  Logging
    # ----------------------------------------------------------------------
    def log_control_state(self, tag: str):
        pacer = self.client_manager.get_pacer_state()
        msg = (
            f"[ctrl] {tag} | round={self.round} clock={self.global_virtual_clock:.2f} "
            f"lr={self.args.learning_rate:.6g} | pacer={pacer}"
        )
        logging.info(msg)

    def log_train_result(self, avg_loss):
        if self.wandb is not None:
            self.wandb.log(
                {
                    "Train/round_to_loss": avg_loss,
                    "Train/round_duration (min)": self.round_duration / 60.0,
                    "Train/client_duration (min)": self.flatten_client_duration,
                    "Train/time_to_round (min)": self.global_virtual_clock / 60.0,
                },
                step=self.round,
            )

    def log_test_result(self):
        perf = self.testing_history["perf"][self.round]
        top1 = perf["top_1"]
        top5 = perf.get("top_5", 0.0)
        loss = perf["loss"]
        clock = perf["clock"]
        self._log_json("TEST_RESULT", {
            "round": int(self.round),
            "clock": float(clock),
            "loss": float(loss),
            "top1": float(top1),
            "top5": float(top5),
        })

        if self.wandb is not None:
            self.wandb.log({"round": self.round, "Agg/top1": top1, "Agg/top5": top5, "Agg/loss": loss})
            self.wandb.log({"clock": clock, "AggWC/top1": top1, "AggWC/top5": top5, "AggWC/loss": loss})

    # ----------------------------------------------------------------------
    #  (De)serialization helpers for RPC
    # ----------------------------------------------------------------------
    def deserialize_response(self, responses):
        """Deserialize the response from executor

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        if responses is None:
            return None
        if isinstance(responses, bytearray):
            responses = bytes(responses)
        if isinstance(responses, str):
            responses = responses.encode("latin1")

        try:
            return pickle.loads(responses)
        except UnicodeDecodeError:
            # Handle payloads containing legacy 8-bit pickled strings.
            return pickle.loads(responses, encoding="latin1")
        except Exception:
            logging.exception(
                "Failed to deserialize response payload of size %s",
                len(responses) if hasattr(responses, "__len__") else "unknown",
            )
            raise

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses, protocol=pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    #  Testing completion
    # ----------------------------------------------------------------------
    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """

        results = results["results"]
        self.test_result_accumulator.append(results)

        if len(self.test_result_accumulator) == len(self.executors):
            self.aggregate_test_result()
            with open(os.path.join(logger.logDir, "testing_perf"), "wb") as fout:
                pickle.dump(self.testing_history, fout)

            logging.info("logging test result")
            self.log_test_result()

            self.broadcast_events_queue.append(commons.START_ROUND)

    # ----------------------------------------------------------------------
    #  Event dispatch / monitor
    # ----------------------------------------------------------------------
    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
            clients (list of int): target client ids for event.

        """
        if clients is None:
            clients = self.sampled_executors
        for client_id in clients:
            self.individual_client_events[client_id].append(event)

    def get_client_conf(self, client_id):
        base_conf = {"learning_rate": self.args.learning_rate}

        if self.args.sample_mode == "pyramidfl":
            ov = self.client_manager.get_pyramidfl_conf(client_id) or {}
            if "local_steps" in ov:
                base_conf["local_steps"] = int(ov["local_steps"])
            base_conf["pyramidfl_dropout_p"] = float(ov.get("dropout_p", 0.0))
            return base_conf

        if not self.args.adaptive_training:
            return base_conf

        meta = self.client_manager.getClient(client_id)
        t_dl = meta.get_download_time(
            cur_time=self.global_virtual_clock, model_size_mb=self.model_size
        )
        train_budget = max(0.0, self.args.t_budget - t_dl)

        trace_pkg = {
            "timestamps_livelab": meta.timestamps_livelab,
            "rate": meta.rate,
            "timestamps_carat": meta.timestamps_carat,
            "availability": meta.availability,
            "batteryLevel": meta.batteryLevel,
            "active": meta.active,
            "inactive": meta.inactive,
            "peak_throughput": meta.peak_throughput,
            "cpu_flops": meta.cpu_flops,
            "gpu_flops": meta.gpu_flops,
        }

        base_conf.update(
            {
                "adaptive_training": True,
                "t_download": t_dl,
                "t_budget_train": train_budget,
                "budget_recheck_steps": self.args.budget_recheck_steps,
                "ewma_lambda": self.args.ewma_lambda,
                "min_payload_frac": self.args.min_payload_frac,
                "run_phase_2": self.args.run_phase_2,
                "start_time": self.global_virtual_clock + t_dl,
                "model_size": self.model_size,
                "model_amount_parameters": self.model_amount_parameters,
                "clock_factor": self.args.clock_factor,
                "dynamic_trace": trace_pkg,
            }
        )
        return base_conf

    def create_client_task(self, executor_id):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executor_id)
        train_config = None
        if next_client_id is not None:
            config = self.get_client_conf(next_client_id)
            padded = {**vars(self.args), **config}
            self.client_conf[next_client_id] = types.SimpleNamespace(**padded)
            train_config = {"client_id": next_client_id, "task_config": config}
        else:
            pass
        return train_config, self.model_wrapper.get_weights()

    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        return {"client_id": client_id}, self.model_wrapper.get_weights()

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): TorchClient id.

        Returns:
            dictionary: Shutdown config for new task.

        """
        return {"client_id": client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.server_events_queue.append((client_id, event, meta, data))

    # ----------------------------------------------------------------------
    #  gRPC handlers
    # ----------------------------------------------------------------------
    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        # Capture peer IP for logging/debugging
        try:
            peer = context.peer()  # e.g., 'ipv4:127.0.0.1:54321' or 'ipv6:[::1]:54321'
            ip = peer
            if peer.startswith('ipv4:'):
                parts = peer.split(':')
                if len(parts) >= 3:
                    ip = parts[1]
            elif peer.startswith('ipv6:'):
                lb = peer.find('['); rb = peer.find(']')
                if lb != -1 and rb != -1 and rb > lb:
                    ip = peer[lb+1:rb]
            self.executor_peers[executor_id] = ip
        except Exception:
            pass
        if executor_id not in self.individual_client_events:
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info("Previous client: %s resumes connecting", executor_id)

        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(
            event=commons.DUMMY_EVENT, meta=dummy_data, data=dummy_data
        )

    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(executor_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg, response_data = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.model_wrapper.get_weights()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg
        ), self.serialize_response(response_data)

        response = job_api_pb2.ServerResponse(
            event=current_event, meta=response_msg, data=response_data
        )
        if current_event != commons.DUMMY_EVENT:
            logging.info("Issue EVENT (%s) to EXECUTOR (%s)", current_event, executor_id)

        return response

    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task."""
        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == commons.CLIENT_TRAIN:
            if execution_status is False:
                logging.error(
                    "Executor %s fails to run client %s, due to %s",
                    executor_id,
                    client_id,
                    execution_msg,
                )

            if (
                self.experiment_mode == commons.SIMULATION_MODE
                and self.resource_manager.has_next_task(executor_id)
            ):
                if commons.CLIENT_TRAIN not in self.individual_client_events[executor_id]:
                    self.individual_client_events[executor_id].append(commons.CLIENT_TRAIN)

        elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(executor_id, event, meta_result, data_result)

        else:
            logging.error("Received undefined event %s from client %s", event, client_id)

        return self.CLIENT_PING(request, context)

    # ----------------------------------------------------------------------
    #  Event loop
    # ----------------------------------------------------------------------
    def event_monitor(self):
        """Activate event handler according to the received new message"""
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST):
                    self.dispatch_client_events(current_event)
                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)
                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle server events
            elif len(self.server_events_queue) > 0:
                (client_id, current_event, meta, data) = self.server_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data)
                    )

                else:
                    logging.error("Event %s is not defined", current_event)

            else:
                time.sleep(0.1)

    # ----------------------------------------------------------------------
    #  Shutdown
    # ----------------------------------------------------------------------
    def stop(self):
        logging.info("Terminating the aggregator ...")
        if self.wandb is not None:
            self.wandb.finish()
        time.sleep(5)


if __name__ == "__main__":
    aggregator = Aggregator(parser.args)
    aggregator.run()
