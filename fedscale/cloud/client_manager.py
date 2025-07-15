import logging
import pickle
from random import Random
from typing import List
import numpy as np
import math

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
            self.ucb_sampler = create_training_selector(args=args)

        if self.mode == 'bliss':
            from thirdparty.bliss.bliss import create_training_selector
            self.bliss_sampler = create_training_selector(args=args)


        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.args = args

        with open(args.clients_file, 'rb') as fin:
            self.clients = pickle.load(fin)
        self.clients_keys = list(self.clients.keys())

    def register_client(self, host_id: int, client_id: int, size: int,
                        duration: float = 1) -> None:
        """Register client information to the client manager.

        Args:
            host_id (int): executor Id.
            client_id (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.

        """
        cd = self.clients[client_id]
        # extract everything your modified ClientMetadata __init__ needs:
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
        )

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(client_id)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                             'duration': duration,
                             }
                self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
            elif self.mode == "bliss":
                feedbacks = {
                    'reward': min(size, self.args.local_steps * self.args.batch_size),
                    'metadata': {
                        'osVersion': cd['osVersion'],
                        'model': cd['model'],
                        'brand': cd['brand'],
                        'os': cd['OS'],
                        'cpu_flops': cd['CPU_FLOPS'],
                        'gpu_flops': cd['GPU_FLOPS'],
                        'internal_memory': cd['internal_memory'],
                        'RAM': cd['RAM'],
                        'peak_throughput': cd['peak_throughput'],
                        'battery': cd['battery']
                    }
                }
                self.bliss_sampler.register_client(client_id, feedbacks)
        else:
            del self.client_metadata[client_id]

    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, client_id):
        return self.client_metadata[client_id]

    def registerDuration(self, client_id, duration):
        if self.mode == "oort":
            self.ucb_sampler.update_duration(client_id, duration)

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
                          duration: float = 1., success: bool = True) -> None:
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

        elif self.mode == "bliss":
            feedbacks = {
                'reward': reward if success else 0,
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

        logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " +
                     f"{len(self.feasibleClients) - len(clients_online)} clients offline")

        return clients_online

    def isClientActive(self, client_id, cur_time):
        return self.client_metadata[client_id].is_active(cur_time)

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


    def select_participants(self, num_of_clients: int, cur_time: float = 0) -> List[int]:
        """Select participating clients for current execution task.

        Args:
            num_of_clients (int): number of participants to select.
            cur_time (float): current wall clock time.

        Returns:
            List[int]: indices of selected clients.

        """
        self.count += 1

        clients_online = self.getOnlineClients(cur_time)

        if len(clients_online) <= num_of_clients:
            return clients_online

        pickled_clients = None
        clients_online_set = set(clients_online)

        if self.mode == "oort" and self.count > 1:
            pickled_clients = self.ucb_sampler.select_participant(
                num_of_clients, feasible_clients=clients_online_set)
            
        elif self.mode == "bliss":

            clients_to_predict_utility = self.bliss_sampler.request_clients_to_predict_utility(clients_online)
            clients_to_refresh_utility = self.bliss_sampler.request_clients_to_refresh_utility(clients_online)

            self.send_bliss_metadata(clients_to_predict_utility, cur_time, self.bliss_sampler.send_clients_to_predict)
            self.send_bliss_metadata(clients_to_refresh_utility, cur_time, self.bliss_sampler.send_clients_to_refresh)

            pickled_clients = self.bliss_sampler.select_participant(num_of_clients)

            self.send_bliss_metadata(pickled_clients, cur_time, self.bliss_sampler.update_client_metadata_pre_training)

        else:
            self.rng.shuffle(clients_online)
            client_len = min(num_of_clients, len(clients_online))
            pickled_clients = clients_online[:client_len]   

        return pickled_clients

    def resampleClients(self, num_of_clients, cur_time=0):
        return self.select_participants(num_of_clients, cur_time)

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucb_sampler.getAllMetrics()
        elif self.mode == "bliss":
            return self.bliss_sampler.getAllMetrics()

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}
