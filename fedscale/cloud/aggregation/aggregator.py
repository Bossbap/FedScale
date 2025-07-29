# -*- coding: utf-8 -*-
import collections
import copy
import math
import os
import pickle
import random
import threading
import time
from concurrent import futures
import types
import pathlib

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

        logging.info(f"Job args {args}")
        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device("cpu")

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.0
        self.round_duration = 0.0
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, "model_" + str(args.this_rank) + ".npy"
        )
        self.last_saved_round = 0

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queues =======
        self.individual_client_events = {}  # Unicast
        self.server_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # ======== runtime information ========
        self.tasks_round = 0
        self.num_of_clients = 0

        self.sampled_participants = []
        self.sampled_executors = []

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

        # ======== Task specific ============
        self.init_task_context()

        # ======== Resume state ============
        self._resume_state = None
        self._restored = False
        if args.resume_from:
            with open(args.resume_from, 'rb') as f:
                self._resume_state = pickle.load(f)
            logging.info("[checkpoint] loaded header of %s", args.resume_from)

    # ----------------------------------------------------------------------
    #  Helpers: snapshot / restore / checkpoint
    # ----------------------------------------------------------------------
    def _snapshot_state(self) -> dict:
        """Return a pickle‑friendly snapshot at the *beginning* of the current round."""
        state = {
            "schema_version": 1,
            "round": self.round,
            "global_clock": self.global_virtual_clock,
            "at_round_begin": True,  # invariant
            "args_state": vars(self.args).copy(),  # mutated args included!
            "model_state": self.model_wrapper.model.state_dict(),
            "client_mgr_state": self.client_manager.get_state(),
            "oort_state": getattr(self.client_manager, "ucb_sampler", None),
            "bliss_state": getattr(self.client_manager, "bliss_sampler", None),
            "optimizer_state": getattr(
                self.model_wrapper.optimizer, "gradient_controller", None
            ),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            # debug / sanity
            "lr_value": float(getattr(self.args, "learning_rate", 0.0)),
            "t_budget": float(getattr(self.args, "t_budget", 0.0)),
        }
        return state

    def _save_checkpoint_if_due(self, tag: str = "") -> None:
        """Persist a checkpoint if the interval matches."""
        if not self.args.checkpoint_interval:
            return
        if self.round % self.args.checkpoint_interval != 0:
            return

        ckpt_path = pathlib.Path(logger.logDir) / f"checkpoint_round{self.round}.pkl"
        # self.log_control_state(f"ckpt.{tag or 'begin'}")
        state = self._snapshot_state()
        # self._quick_digest("before-save")
        with ckpt_path.open("wb") as f:
            pickle.dump(state, f)
        logging.info("[checkpoint] saved %s", ckpt_path)

    def _restore_from_checkpoint(self, st: dict) -> None:
        """Restore everything from a previously saved checkpoint."""
        # 1) Args (mutated runtime config)
        args_state = st.get("args_state", None)
        if args_state is not None:
            for k, v in args_state.items():
                if hasattr(self.args, k):
                    setattr(self.args, k, v)

        # 2) Round / clock
        self.round = st["round"]
        self.global_virtual_clock = st["global_clock"]

        # 3) Model
        model_state = st.get("model_state")
        if model_state is not None:
            self.model_wrapper.model.load_state_dict(model_state)

        # 4) Optimizer server-side controller (e.g., Yogi)
        opt_state = st.get("optimizer_state")
        if opt_state:
            self.model_wrapper.optimizer.gradient_controller.__dict__.update(
                opt_state.__dict__
            )

        # 5) Samplers
        if st.get("oort_state") and hasattr(self.client_manager, "ucb_sampler"):
            self.client_manager.ucb_sampler.__dict__.update(
                st["oort_state"].__dict__
            )
        if st.get("bliss_state") and hasattr(self.client_manager, "bliss_sampler"):
            self.client_manager.bliss_sampler.__dict__.update(
                st["bliss_state"].__dict__
            )

        # 6) Client manager
        if st.get("client_mgr_state"):
            self.client_manager.load_state(st["client_mgr_state"])

        # 7) RNGs
        rng = st.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])

        self._restored = True
        # self._quick_digest("after-restore")


    def _optimizer_digest(self):
        gc = getattr(self.model_wrapper.optimizer, "gradient_controller", None)
        if gc is None:
            return {}
        # Log a few key fields if they exist
        out = {}
        for k in ["t", "m", "v", "v_hat", "lr", "beta1", "beta2", "tau"]:
            if hasattr(gc, k):
                v = getattr(gc, k)
                # vectors can be huge; just hash or show first 3 elems
                if isinstance(v, (list, tuple)) and len(v) > 3:
                    out[k] = (v[:3], "...", len(v))
                else:
                    out[k] = v
        return out

    def _quick_digest(self, tag):
        logging.info(
            "[ckpt-digest %s] round=%d clock=%.2f lr=%.6g pacer=%s opt=%s",
            tag, self.round, self.global_virtual_clock, self.args.learning_rate,
            self.client_manager.get_pacer_state(),
            self._optimizer_digest()
        )


    # ----------------------------------------------------------------------
    #  Basic env / comms
    # ----------------------------------------------------------------------
    def setup_env(self):
        """Set up experiments environment and server optimizer"""
        self.setup_seed(seed=1)

    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

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

        # ==== restore if requested ====
        if self._resume_state and not self._restored:
            self._restore_from_checkpoint(self._resume_state)

        self.model_weights = self.model_wrapper.get_weights()

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

        # sample_mode: random, oort or bliss

        return ClientManager(args.sample_mode, args=args)

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

        self.tasks_round = len(kept)
        self.round_duration = max(r["wall_duration"] for r in kept) if kept else 0.0
        self.flatten_client_duration = np.array([r["wall_duration"] for r in kept])
        self.virtual_client_clock = {
            r["client_id"]: r["wall_duration"] for r in self.pending_client_results
        }

        self.model_in_update = 0
        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        avg_util = (sum(r["utility"] for r in kept) / max(1, len(kept))) if kept else 0.0

        for r in kept:
            cid = r["client_id"]

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
                r["client_id"],
                avg_util,
                time_stamp=self.round,
                duration=r["wall_duration"],
                success=False,
            )

        self.round_stragglers = []
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
        logging.info(
            "Received executor %s information, %d/%d",
            executorId,
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

        if self.experiment_mode == commons.SIMULATION_MODE:
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                roundDuration = self.client_manager.get_completion_time(
                    client_to_run,
                    cur_time=self.global_virtual_clock,
                    batch_size=client_cfg.batch_size,
                    local_steps=client_cfg.local_steps,
                    model_size=self.model_size,
                    model_amount_parameters=self.model_amount_parameters,
                )

                if self.client_manager.mode == "oort":
                    self.client_manager.registerDuration(client_to_run, duration=roundDuration)

                if self.client_manager.isClientActive(
                    client_to_run, roundDuration + self.global_virtual_clock
                ):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = roundDuration

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
            completionTimes.sort()

            return (
                clients_to_run,
                stragglers,
                completed_client_clock,
                round_duration,
                completionTimes[:num_clients_to_collect],
            )

        else:
            completed_client_clock = {
                client: {"computation": 1, "communication": 1} for client in sampled_clients
            }
            completionTimes = [1 for _ in sampled_clients]
            return (
                sampled_clients,
                sampled_clients,
                completed_client_clock,
                1,
                completionTimes,
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
        cid = results["client_id"]

        if self.args.adaptive_training and results.get("wall_duration", 0) > 0:
            actual_dur = results["wall_duration"]
            self.client_manager.registerDuration(cid, actual_dur)
            dur_for_feedback = actual_dur
        else:
            dur_for_feedback = self.virtual_client_clock[cid]

        logging.info(
            "[aggregator] client %s finished, simulated_duration = %.2f s",
            cid,
            dur_for_feedback,
        )

        self.client_manager.register_feedback(
            cid,
            results["utility"],
            time_stamp=self.round,
            duration=dur_for_feedback,
        )

        with self.update_lock:
            self.model_in_update += 1
            self.update_weight_aggregation(results)

    def update_weight_aggregation(self, results):

        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """

        upd = results["update_weight"]

        if self._is_first_result_in_round():
            if isinstance(upd, dict):
                self.param_order = list(upd.keys())
                self.model_weights = {k: v.copy() for k, v in upd.items()}
            else:
                self.param_order = list(range(len(upd)))
                self.model_weights = [w.copy() for w in upd]
            return

        if isinstance(upd, dict):
            for k, v in upd.items():
                self.model_weights[k] += v
        else:
            for i, v in enumerate(upd):
                self.model_weights[i] += v

        if self._is_last_result_in_round():
            denom = float(self.tasks_round)

            if isinstance(self.model_weights, dict):
                to_send = []
                for k in self.param_order:
                    v = self.model_weights[k]
                    v = v.astype(np.float32) if np.issubdtype(v.dtype, np.integer) else v
                    self.model_weights[k] = v / denom
                    to_send.append(self.model_weights[k])
            else:
                to_send = [w / denom for w in self.model_weights]

            self.model_wrapper.set_weights(
                copy.deepcopy(to_send),
                client_training_results=self.client_training_results,
            )

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
        bootstrap = (self.tasks_round == 0 and len(self.stats_util_accumulator) == 0)

        # ------------------------------------------------------------------
        # 0) BOOTSTRAP / RESUME
        # ------------------------------------------------------------------
        if bootstrap:
            if getattr(self, "_restored", False):
                logging.info("[checkpoint] Resuming at the BEGINNING of round %d", self.round)
            else:
                logging.info("Bootstrap: BEGINNING of round %d", self.round)

            # Optionally checkpoint the very beginning of this round
            self._save_checkpoint_if_due(tag="bootstrap")

            # === Select & schedule THIS round ===
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
            else:
                (
                    clients_to_run,
                    round_stragglers,
                    virtual_client_clock,
                    round_duration,
                    flatten_client_duration,
                ) = self.tictak_client_tasks(
                    self.sampled_participants, self.args.num_participants
                )
                self.tasks_round = len(clients_to_run)
                self.virtual_client_clock = virtual_client_clock
                self.flatten_client_duration = np.array(flatten_client_duration)
                self.round_duration = round_duration
                self.round_stragglers = round_stragglers

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
        logging.info(
            "*** ROUND %d COMPLETE: got %d/%d updates, moving on ***",
            self.round,
            len(self.stats_util_accumulator),
            self.tasks_round,
        )

        # Advance wall-clock & bump round number
        self.global_virtual_clock += self.round_duration
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
                time_stamp=self.round,
                duration=self.virtual_client_clock[client_id],
                success=False,
            )

        # Log loss / training stats
        avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
        logging.info(
            "Wall clock: %ds, round: %d, Planned participants: %d, Succeed: %d, Loss: %f",
            round(self.global_virtual_clock),
            self.round,
            len(self.sampled_participants),
            len(self.stats_util_accumulator),
            avg_loss,
        )
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # --- Apply LR decay (& any other beginning-of-round mutations) ---
        self.update_default_task_config()

        # --- Checkpoint NOW (beginning of round self.round) --------------
        self._save_checkpoint_if_due(tag="begin-r")

        # ======= Now select and dispatch the *next* round =======
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
        else:
            (
                clients_to_run,
                round_stragglers,
                virtual_client_clock,
                round_duration,
                flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_participants, self.args.num_participants
            )

            self.tasks_round = len(clients_to_run)
            self.virtual_client_clock = virtual_client_clock
            self.flatten_client_duration = np.array(flatten_client_duration)
            self.round_duration = round_duration
            self.round_stragglers = round_stragglers

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

        if self.wandb is not None:
            self.wandb.log({"round": self.round, "Agg/top1": top1, "Agg/top5": top5, "Agg/loss": loss})
            self.wandb.log({"clock": clock, "AggWC/top1": top1, "AggWC/top5": top5, "AggWC/loss": loss})

    def save_model(self):
        if parser.args.save_checkpoint and self.last_saved_round < self.round:
            self.last_saved_round = self.round
            np.save(self.temp_model_path, self.model_weights)
            if self.wandb is not None:
                artifact = self.wandb.Artifact(name="model_" + str(self.this_rank), type="model")
                artifact.add_file(local_path=self.temp_model_path)
                self.wandb.log_artifact(artifact)

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
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses)

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

            self.save_model()

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