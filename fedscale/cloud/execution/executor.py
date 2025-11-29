# -*- coding: utf-8 -*-
import collections
import gc
import pickle
import random
import time
from argparse import Namespace
import csv
import os

# Avoid gRPC fork warnings when subprocesses/threads are active.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_PYTHON_FORK_SUPPORT_ENABLED", "0")

import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.channels.channel_context import ClientConnections
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.adaptive_torch_client import AdaptiveTorchClient
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.fllibs import *
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset


class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # initiate the executor log path, and executor ips
        logger.initiate_client_setting()

        self.model_adapter = self.get_client_trainer(args).get_model_adapter(
            init_model()
        )

        self.args = args
        self.base_seed = getattr(args, "sample_seed", None)
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.training_sets = self.test_dataset = None

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"executor{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
            else:
                logging.error("Warning: wandb has already been initialized")

        else:
            self.wandb = None
        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment"""
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        base = self.base_seed if self.base_seed is not None else 233
        per_executor_seed = (int(base) + int(self.this_rank)) % (2**32)
        self.setup_seed(seed=per_executor_seed)

    def setup_communication(self):
        """Set up grpc connection"""
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=None):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        if seed is None:
            seed = 233
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)"""
        pass

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == "nlp":
            self.collate_fn = collate
        elif self.args.task == "voice":
            self.collate_fn = voice_collate_fn
        # load data partitionxr (entire_train_data)
        logging.info(f"Data partitioner starts ...\nNumber of Paricipants = {self.args.num_participants}")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class
        )
        # count how many real clients appear in the CSV so every one
        # gets its own shard (1-to-1 mapping)
        with open(self.args.data_map_file) as f:
            n_real_clients = len({row.split(',')[0] for idx, row in enumerate(f) if idx})

        training_sets.partition_data_helper(
            num_clients=n_real_clients,            # purely cosmetic – ignored by trace loader
            data_map_file=self.args.data_map_file,
        )

        testing_sets = DataPartitioner(
            data=test_dataset,
            args=self.args,
            numOfClass=self.args.num_class,
            isTest=True,
        )
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message."""
        self.setup_env()
        try:
            self.training_sets, self.testing_sets = self.init_data()
        except Exception:
            logging.exception("[executor %s] Fatal error during init_data (dataset build/preproc)", self.executor_id)
            raise
        try:
            self.setup_communication()
        except Exception:
            logging.exception("[executor %s] Fatal error setting up communication", self.executor_id)
            raise
        try:
            self.event_monitor()
        except Exception:
            logging.exception("[executor %s] Unhandled exception in event monitor", self.executor_id)
            raise

    def dispatch_worker_events(self, request):
        """Add new events to worker queues

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.

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
            responses (string, bool, or bytes): TorchClient responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.

        """
        return pickle.dumps(responses, protocol=pickle.HIGHEST_PROTOCOL)

    def UpdateModel(self, model_weights):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config

        """
        self.round += 1
        self.model_adapter.set_weights(model_weights, is_aggregator=False)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config = config["client_id"], config["task_config"]

        if "model" not in config or not config["model"]:
            raise ValueError("The 'model' object must be provided and non-null in the training config.")
        client_conf = self.override_conf(train_config)
        try:
            train_res = self.training_handler(
                client_id=client_id, conf=client_conf, model=config["model"]
            )
        except Exception:
            # Build a well-formed failure payload to keep the aggregator robust.
            # Fall back to reporting zero utility and the current model weights,
            # so aggregation can proceed without KeyErrors or type mismatches.
            logging.exception(
                "[executor %s] Uncaught exception, cid=%d",
                self.executor_id,
                client_id,
            )
            try:
                # Ensure we always return a dict-of-arrays consistent with TorchClient
                # so the aggregator's mixer can handle it.
                model_obj = config.get("model", None)
                if model_obj is not None and hasattr(model_obj, "state_dict"):
                    sd = model_obj.state_dict()
                    update_weight = {k: v.detach().cpu().numpy() for k, v in sd.items()}
                else:
                    update_weight = {}
            except Exception:
                update_weight = {}

            train_res = {
                "client_id": client_id,
                "moving_loss": 0.0,
                "trained_size": 0,
                "success": False,
                "utility": 0.0,
                "update_weight": update_weight,
                "wall_duration": 0.0,
                "iters": 0,
            }


        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=str(client_id),
                executor_id=self.executor_id,
                event=commons.CLIENT_TRAIN,
                status=True,
                msg=None,
                meta_result=None,
                data_result=None,
            )
        )
        self.dispatch_worker_events(response)

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        # Remember which real client id the server asked us to evaluate,
        # so the testing handler can load the matching shard when available.
        try:
            self._test_client_id = int(config.get("client_id", self.this_rank))
        except Exception:
            self._test_client_id = self.this_rank
        test_res = self.testing_handler(model=config["model"])
        test_res = {"executorId": self.this_rank, "results": test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id,
                executor_id=self.executor_id,
                event=commons.MODEL_TEST,
                status=True,
                msg=None,
                meta_result=None,
                data_result=self.serialize_response(test_res),
            )
        )
        self.dispatch_worker_events(response)

    def Stop(self):
        """Stop the current executor"""
        logging.info(f"Terminating the executor ...")
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True
        if self.wandb != None:
            self.wandb.finish()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        client_dict = self.training_sets.client_dict

        client_ids = []
        sizes      = []
        for cid, idx_list in client_dict.items():
            client_ids.append(int(cid))        # make sure they are ints
            sizes.append(len(idx_list))

        return {
            "client_ids": client_ids,
            "size"      : sizes
        }

    def override_conf(self, config):
        """Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        if conf.engine == commons.TENSORFLOW:
            return TensorflowClient(conf)
        elif conf.engine == commons.PYTORCH:
            if conf.task == "rl":
                return RLClient(conf)
            else:
                if conf.adaptive_training:
                    return AdaptiveTorchClient(conf)
                else:
                    return TorchClient(conf)
        raise NotImplementedError("Currently, FedScale supports only TensorFlow and PyTorch engines.")

    def training_handler(self, client_id, conf, model):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        self.model_adapter.set_weights(model, is_aggregator=False)
        conf.client_id = client_id
        # Always source tokenizer from fllibs at runtime to avoid stale None from star-import.
        try:
            import fedscale.cloud.fllibs as fllibs
            conf.tokenizer = getattr(conf, 'tokenizer', None) or getattr(fllibs, 'tokenizer', None)
        except Exception:
            conf.tokenizer = None
        client_data = (
            self.training_sets
            if self.args.task == "rl"
            else select_dataset(
                client_id,
                self.training_sets,
                batch_size=conf.batch_size,
                args=self.args,
                collate_fn=self.collate_fn,
            )
        )
        client = self.get_client_trainer(self.args)
        train_res = client.train(
            client_data=client_data, model=self.model_adapter.get_model(), conf=conf
        )

        return train_res

    def testing_handler(self, model):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        self.model_adapter.set_weights(model, is_aggregator=False)
        # Pull live tokenizer from fllibs to pass into testing path
        try:
            import fedscale.cloud.fllibs as fllibs
            live_tok = getattr(fllibs, 'tokenizer', None)
        except Exception:
            live_tok = None
        test_config = self.override_conf({
            # For CSV-based partitions, the rank is the real client id
            # whose test shard will be loaded below.
            "rank": getattr(self, "_test_client_id", self.this_rank),
            "memory_capacity": self.args.memory_capacity,
            "tokenizer": live_tok,
        })
        client = self.get_client_trainer(test_config)
        # When a mapping CSV is available, use the real client id to pick
        # the test subset; otherwise fall back to executor rank.
        select_rank = getattr(self, "_test_client_id", self.this_rank)
        data_loader = select_dataset(
            select_rank,
            self.testing_sets,
            batch_size=self.args.test_bsz,
            args=self.args,
            isTest=True,
            collate_fn=self.collate_fn,
        )

        test_results = client.test(
            data_loader, model=self.model_adapter.get_model(), conf=test_config
        )
        gc.collect()

        return test_results

    def client_register(self):
        """Register the executor information to the aggregator"""
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler()
                        ),
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                # Connection retries are expected during orchestrator startup; keep log noise low.
                logging.info(
                    f"Failed to connect to aggregator {e}. Will retry in 5 sec."
                )
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task"""
        response = self.aggregator_communicator.stub.CLIENT_PING(
            job_api_pb2.PingRequest(
                client_id=self.executor_id, executor_id=self.executor_id
            )
        )
        # logging.info("[executor %s] ping server (round=%d, queue=%d)",
        #       self.executor_id, self.round, len(self.event_queue))
        self.dispatch_worker_events(response)

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")
        self.client_register()

        while not self.received_stop_request:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config["model"] = train_model
                    train_config["client_id"] = int(train_config["client_id"])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(
                            client_id=str(client_id),
                            executor_id=self.executor_id,
                            event=commons.UPLOAD_MODEL,
                            status=True,
                            msg=None,
                            meta_result=None,
                            data_result=self.serialize_response(train_res),
                        )
                    )
                    future_call.add_done_callback(
                        lambda _response: self.dispatch_worker_events(
                            _response.result()
                        )
                    )

                elif current_event == commons.MODEL_TEST:
                    test_config = self.deserialize_response(request.meta)
                    test_model = self.deserialize_response(request.data)
                    test_config["model"] = test_model
                    test_config["client_id"] = int(test_config["client_id"])
                    self.Test(test_config)

                elif current_event == commons.UPDATE_MODEL:
                    model_weights = self.deserialize_response(request.data)
                    self.UpdateModel(model_weights)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                try:
                    self.client_ping()
                except Exception as e:
                    logging.info(
                        f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ..."
                    )
                    self.Stop()


if __name__ == "__main__":
    try:
        executor = Executor(parser.args)
        executor.run()
    except Exception:
        # Ensure a clear traceback is emitted in executor logs and process exits non-zero
        logging.exception("[executor %s] Uncaught top-level exception", getattr(parser.args, 'this_rank', 'unknown'))
        import sys
        sys.exit(1)
