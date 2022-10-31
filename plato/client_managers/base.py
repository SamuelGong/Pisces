import copy
import logging
import os
import time

import numpy as np
import requests
import random
import pickle
import asyncio
import gc
from plato.config import Config
from plato.client_managers.oort import create_training_selector as create_oort_selector


def my_random_zipfian(a, n, amin, amax):
    prob = np.array([1 / k**a for k
                     in np.arange(1, n + 1)])
    res = [(e - min(prob)) / (max(prob) - min(prob)) * (amax - amin) + amin for e in prob]
    res = [round(e, 2) for e in res]
    np.random.shuffle(res)
    return res


class ClientManager:
    """The base class for federated learning client managers."""
    def __init__(self):
        self.global_start_time = None
        self.trainset_sizes = {}
        self.total_clients = Config().clients.total_clients

        self.client_selector = None
        if hasattr(Config().server, 'client_selection') \
                and Config().server.client_selection.name == 'oort':
            params = Config().server.client_selection.parameters
            self.client_selector = create_oort_selector(params)

        self.client_train_dict = {
            client_id: []
            for client_id in range(1, self.total_clients + 1)
        }
        self.used_models = {}
        self.used_models_clients_map = {}
        self.server_test_list = []

        self.response_latencies = self.init_resp_lat()
        self.updated_concentration = {}
        self.updated_partition_size = {}

        self.agg_time = []
        self.start_time = time.perf_counter()
        self.expected_corrupted_clients = self.init_corruption_portion()
        self.detected_corrupted_clients = []

        if hasattr(Config().data, 'concentration_update'):
            self.set_updated_concentration()

        if hasattr(Config().data, 'partition_size_update'):
            self.set_updated_partition_size()

    def set_updated_partition_size(self):
        for client_id, v in self.response_latencies.items():
            scale_const = Config().data.partition_size_update
            # stragglers have more data
            self.updated_partition_size[client_id] = \
                max(1, int(scale_const * v['expected']))

        logging.info(f"Updated partition_size: {self.updated_partition_size}")

    def get_updated_partition_size(self, client_id):
        return self.updated_partition_size[client_id]

    def set_updated_concentration(self):
        for client_id, v in self.response_latencies.items():
            scale_const = Config().data.concentration_update
            # stragglers are also balanced data owners
            self.updated_concentration[client_id] = \
                scale_const * v['expected']

        logging.info(f"Updated concentration: {self.updated_concentration}")

    def get_updated_concentration(self, client_id):
        return self.updated_concentration[client_id]

    def get_corruption_simulation_plan(self):
        return self.expected_corrupted_clients

    def init_corruption_portion(self):
        result_list = []
        if hasattr(Config().data, "data_corruption") \
                and Config().data.data_corruption.type == "flip":
            type = Config().data.data_corruption.type
            portion = Config().data.data_corruption.portion
            seed = Config().data.data_corruption.seed
            np.random.seed(seed)

            corrupted_num_clients = max(1, int(portion * self.total_clients))
            corrupted_clients = np.random.choice(list(range(1, 1 + self.total_clients)),
                                                 corrupted_num_clients,
                                                 replace=False).tolist()
            corrupted_clients = sorted(corrupted_clients)
            logging.info(f"Simulating corrupted clients "
                         f"with method {type}: {corrupted_clients}.")
            result_list = corrupted_clients

        return result_list

    def init_resp_lat(self):
        result_dict = {}

        if hasattr(Config().server, 'response_latency_distribution'):
            name = Config().server.response_latency_distribution.name
            args = Config().server.response_latency_distribution.args

            if name == 'normal':
                mean, std, seed = args.mean, args.std, args.seed
                np.random.seed(seed)
                expected_latencies = np.random.normal(mean, std,
                                                      self.total_clients)
            elif name == 'zeta':
                a, loc, seed = args.a, args.loc, args.seed
                np.random.seed(seed)
                expected_latencies = np.random.zipf(a,
                                                    self.total_clients) + loc
            elif name == 'zipf':
                a, amin, amax, seed = args.a, args.min, args.max, args.seed
                np.random.seed(seed)
                expected_latencies = my_random_zipfian(
                    a, self.total_clients, amin, amax)
            elif name == 'customized':
                comp_scaling_factor, comm_scaling_factor, trace_url, seed = \
                    args.comp_scaling_factor, args.comm_scaling_factor, \
                    args.trace_url, args.seed
                trace_folder = './customized_trace'
                if not os.path.isdir(trace_folder):
                    os.makedirs(trace_folder)
                trace_path = os.path.join(trace_folder,
                                          trace_url.split('/')[-1])

                if not os.path.isfile(trace_path):
                    res = requests.get(trace_url, stream=True)
                    with open(trace_path, "wb+") as file:
                        for chunk in res.iter_content(chunk_size=1024):
                            file.write(chunk)
                            file.flush()

                with open(trace_path, 'rb') as fin:
                    trace = pickle.load(fin)

                trace_client_ids = list(trace.keys())
                random.seed(seed)
                sampled_clients = random.sample(trace_client_ids,
                                                self.total_clients)
                expected_latencies = [[
                    trace[client_id]['computation'] * 0.001 *
                    comp_scaling_factor,
                    trace[client_id]['communication'] * comm_scaling_factor
                ] for client_id in sampled_clients]
                # resulting item[0]: speed of processing one data sample in secs.
                # resulting item[1]: network throughput in Kibits/s
            else:
                logging.info(
                    f"Unknown distribution name (%s) for "
                    f"simulating resource heterogeneity!", name)
                raise ValueError

            logging.info(
                "Simulating resource heterogeneity for "
                "%d clients. The calculated expected latencies are %s.",
                self.total_clients, expected_latencies)
            for idx, expected_latency in enumerate(expected_latencies):
                client_id = idx + 1
                result_dict[client_id] = {'expected': expected_latency}
        else:
            for idx in range(self.total_clients):
                client_id = idx + 1
                result_dict[client_id] = {}

        return result_dict

    def update_resp_lat(self, name, **kwargs):
        if name == "customized":
            trainset_size = kwargs['trainset_size']
            client_id = kwargs['client_id']
            epochs = Config().trainer.epochs
            total_samples = epochs * trainset_size
            latency_per_sample = self.response_latencies[client_id][
                'expected'][0]
            computing_latency = total_samples * latency_per_sample
            self.response_latencies[client_id]['expected'][
                0] = computing_latency

    async def simulate_resp_lat(self, client_id, payload_size, actual_latency):
        expected_latency = self.response_latencies[client_id]['expected']

        if isinstance(expected_latency, list):  # name == customized
            computing_latency = expected_latency[0]
            network_bandwidth = expected_latency[1]
            communication_latency = 2 * payload_size * 8 / 1024 / network_bandwidth
            expected_latency = computing_latency + communication_latency

        wait_time = expected_latency - actual_latency
        if wait_time > 0:
            logging.info(
                f"[Server #%d] Simulating expected latency (%.2f s) "
                f"by elongating client #%d's latency for %.2f s.", os.getpid(),
                expected_latency, client_id, wait_time)
            await asyncio.sleep(wait_time)  # yield to others
            actual_latency = expected_latency
        else:
            logging.info(
                f"[Server #%d] Client #%d's actual latency (%.2f s) exceeds "
                f"the expected one (%.2f s). Consider adjusting "
                f"the used distribution for simulation accordingly.",
                os.getpid(), client_id, actual_latency, expected_latency)
        return actual_latency

    def init_client_selector(self, name, **kwargs):
        if name == "oort":
            client_id = kwargs["client_id"]
            feedbacks = kwargs["feedbacks"]
            self.client_selector.register_client(client_id,
                                                 feedbacks=feedbacks)

    def update_client_selector(self, name, **kwargs):
        if name == "oort":
            client_id = kwargs['client_id']
            time_stamp = kwargs['time_stamp']
            utility = kwargs['utility']
            duration = kwargs['duration']
            reward = self.trainset_sizes[client_id] * utility
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'time_stamp': time_stamp,
                'status': True,
            }
            self.client_selector.update_client_util(client_id,
                                                    feedbacks=feedbacks)
            logging.info(f"Updated feedbacks for client {client_id}: utility {utility}, "
                         f"reward: {reward}, duration {duration}.")

    def choose_clients(self, clients_pool, sample_size):
        if self.client_selector:
            result = self.client_selector.select_participant(
                num_of_clients=sample_size, feasible_clients=clients_pool)
        else:
            result = random.sample(clients_pool, sample_size)

        return result

    def set_global_start_timestamp(self, start_timestamp):
        self.global_start_time = start_timestamp

    def get_start_timestamp(self, client_id):
        training_records = self.client_train_dict[client_id]
        return training_records[-1]["begin_time"]

    def record_training_start(self, client_id,
                                begin_time, model_version):
        self.client_train_dict[client_id].append({
            "begin_time": begin_time,
            "start_version": model_version
        })

    def record_training_end(self, client_id, end_time):
        self.client_train_dict[client_id][-1].update({
            "end_time": end_time
        })

    def record_used_global_model(self, client_id,
                                 model_version, model):
        if model_version not in self.used_models:
            model_copy = copy.deepcopy(model)
            self.used_models_clients_map[model_version] = [client_id]
            self.used_models[model_version] = model_copy
        else:
            self.used_models_clients_map[model_version].append(client_id)

    def get_used_global_model(self, model_version):
        return self.used_models[model_version]

    def cleanup_used_global_model(self, client_id, model_version):
        self.used_models_clients_map[model_version].remove(client_id)
        if not self.used_models_clients_map[model_version]:
            del self.used_models[model_version]
        gc.collect()

    def record_aggregation(self, client_id_list, agg_time, model_version):
        for client_id in client_id_list:
            training_records = self.client_train_dict[client_id]

            # assuming that the last update associated with a client
            # is the one that is taken to aggregate
            for idx in range(len(training_records) - 1, -1, -1):
                if "end_time" in training_records[idx]:
                    break

            self.client_train_dict[client_id][idx].update({
                "agg_version": model_version,
                "agg_time": agg_time
            })

        self.agg_time.append((agg_time, model_version))

    def get_done_client_latest_start_version(self, client_id):
        training_records = self.client_train_dict[client_id]
        for idx in range(len(training_records) - 1, -1, -1):
            if "end_time" in training_records[idx]:
                break

        start_version = training_records[idx]["start_version"]
        return start_version

    def get_async_client_lag(self, client_id, current_version=None, running=False, multiple=None):
        training_records = self.client_train_dict[client_id]
        found = False
        if current_version is None:  # fetching latest record
            if multiple is not None and isinstance(multiple, int):  # for moving average
                n = multiple
                result = []
                for idx in range(len(training_records) - 1, -1, -1):
                    if "agg_version" in training_records[idx]:  # the most recent aggregated version
                        agg_version = training_records[idx]["agg_version"]
                        start_version = training_records[idx]["start_version"]
                        result.append(agg_version - start_version)
                        if len(result) == n:
                            break
                return result
            else:  # only a scalar
                for idx in range(len(training_records) - 1, -1, -1):
                    if "agg_version" in training_records[idx]:  # the most recent aggregated version
                        found = True
                        break

                if found:
                    agg_version = training_records[idx]["agg_version"]
                    start_version = training_records[idx]["start_version"]
                    return agg_version - start_version
        else:  # compute current value'
            if not running:  # for finished clients
                idx = len(training_records) - 1
                if len(training_records) > 0:
                    while "end_time" not in training_records[idx]:  # the most recent not aggregated yet buffered version
                        idx -= 1
                        if idx == -1:
                            break

                    if idx > -1:
                        found = True
            else:  # for running clients
                for idx in range(len(training_records) - 1, -1, -1):
                    if "start_version" in training_records[idx]:  # the most recent running version
                        found = True
                        break

            if found:
                start_version = training_records[idx]["start_version"]
                return current_version - start_version

        # return a safe and optimistic answer
        return 0

    def get_async_delayed_clients(self, current_version, selected_clients,
                                  done_clients):
        lag_bound = Config().server.asynchronous.staleness_bound
        delayed_clients = []
        for client_id in selected_clients:
            lag = self.get_async_client_lag(client_id, current_version, running=True)
            if lag >= lag_bound and client_id not in done_clients:
                delayed_clients.append(client_id)
        return delayed_clients

    def set_trainset_size(self, client_id, length):
        self.trainset_sizes[client_id] = length

    def staleness_factor_calculator(self, lag, agg_version):
        # popular setting
        return 1.0 / pow(lag + 1, 0.5)
