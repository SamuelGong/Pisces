import time
import logging
import numpy as np
import random
from plato.client_managers import base
from plato.config import Config
from sklearn.cluster import DBSCAN
EPSILON = 1e-8


class ClientManager(base.ClientManager):
    def __init__(self):
        super().__init__()
        self.client_per_round = Config().clients.per_round
        self.seconds_per_step = Config().server.asynchronous.seconds_per_step

        # main data structures
        self.utility = {
            client_id: 0
            for client_id in range(1, self.total_clients + 1)
        }
        self.norm = {
            client_id: 0
            for client_id in range(1, self.total_clients + 1)
        }

        self.explored_clients = []
        self.clients_to_select = []

        params = Config().server.asynchronous.sirius
        self.explore_factor = params.explore_factor
        self.exploration_decaying_factor = params.exploration_decaying_factor
        self.min_explore_factor = params.min_explore_factor
        self.staleness_penalty_factor = params.staleness_penalty_factor
        self.speed_penalty_factor = params.speed_penalty_factor \
            if hasattr(params, "speed_penalty_factor") else 0.5

        self.last_response_latency = {
            client_id: 1.0  # TODO: change if necessary
            for client_id in range(1, self.total_clients + 1)
        }
        self.independent_selection = params.independent_selection \
            if hasattr(params, "independent_selection") else False
        self.staleness_bound = self.client_per_round
        self.staleness_bound_factor = params.staleness_bound_factor \
            if hasattr(params, "staleness_bound_factor") else 0.5
        self.statistical_only = False
        if hasattr(params, "statistical_only") \
                and params.statistical_only == True:
            self.statistical_only = True
        self.agg_interval = None
        self.bounded_staleness = params.bounded_staleness \
            if hasattr(params, "bounded_staleness") else False
        self.staleness_discounted = params.staleness_discounted \
            if hasattr(params, "staleness_discounted") else True
        self.robustness = params.robustness \
            if hasattr(params, "robustness") else False
        self.model_versions_clients_dict = {}
        self.reliability_credit_record = {
            client_id: 5 # TODO: avoid hard-coding
            for client_id in range(1, self.total_clients + 1)
        }

        self.version = 1
        if hasattr(params, "version"):
            self.version = params.version

        if self.version == 1:  # amortized testing accuracy gain
            self.acc_gain_factor = 2.8
            if hasattr(params, "acc_gain_factor"):
                self.acc_gain_factor = params.acc_gain_factor
            self.acc_gain_decay = 1.0
            if hasattr(params, "acc_gain_decay"):
                self.acc_gain_decay = params.acc_gain_decay
        elif self.version == 2:  # norm of local model updates
            self.utility_decay = 0.2
            if hasattr(params, "utility_decay"):
                self.utility_decay = params.utility_decay

        if hasattr(params, "client_selection"):
            self.client_selection = params.client_selection
        else:
            self.client_selection = "principled"
        if hasattr(params, "threshold_aggregation"):
            self.aggregation_threshold = max(1, int(np.floor(
                params.threshold_aggregation * self.client_per_round
            )))
        else:
            self.aggregation_threshold = None

        self.seed = params.seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Should work for whatever figures. Only for the first selection
        self.done_first_selection = False

    def get_version(self):
        return self.version

    def detect_outliers(self, tuples):
        start_time = time.perf_counter()

        client_id_list = [tu[0] for tu in tuples]
        loss_list = [tu[1] for tu in tuples]
        loss_list = np.array(loss_list).reshape(-1, 1)
        min_samples = self.client_per_round // 2  # TODO: avoid hard-coding
        eps = 0.5
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loss_list)
        result = clustering.labels_.tolist()
        outliers = [client_id_list[idx]
                    for idx, e in enumerate(result) if e == -1]
        debug_dict = {
            'client_id_list': client_id_list,
            'loss_list': loss_list.squeeze(-1),  # for ease of reading
            'DBSCAN_res': result
        }
        logging.info(f"[Debug] debug_dict for DBSCAN: {debug_dict}.")
        logging.info(f"[Debug] Note actual outliers: {self.expected_corrupted_clients}.")

        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        logging.info(f"Outliers detected by DBSCAN "
                     f"in {duration} sec: {outliers}.")

        newly_detected_outliers = []
        for client_id in outliers:
            self.reliability_credit_record[client_id] -= 1
            if client_id not in self.detected_corrupted_clients:
                current_credit = self.reliability_credit_record[client_id]
                if current_credit == 0:
                    self.detected_corrupted_clients.append(client_id)
                    newly_detected_outliers.append(client_id)

        if len(newly_detected_outliers) == 0:
            logging.info(f"No new outliers.")
        else:
            newly_detected_outliers = sorted(newly_detected_outliers )
            logging.info(f"{len(newly_detected_outliers)} clients "
                         f"are newly taken as outliers"
                         f": {newly_detected_outliers}.")

    def update_utilities_using_loss(self, client_id, loss_norm):
        debug_dict = {}

        if client_id not in self.explored_clients:
            self.explored_clients.append(client_id)

        training_records = self.client_train_dict[client_id]
        idx = -1
        while "end_time" not in training_records[idx]:
            idx -= 1
        latest_record_tuple = training_records[idx]
        training_begin_time = latest_record_tuple["begin_time"]
        training_end_time = latest_record_tuple["end_time"]
        response_latency = training_end_time - training_begin_time
        self.last_response_latency[client_id] = response_latency

        if self.robustness:
            start_version = latest_record_tuple["start_version"]
            if start_version not in self.model_versions_clients_dict:
                self.model_versions_clients_dict[start_version] = [(client_id, loss_norm)]
            else:
                self.model_versions_clients_dict[start_version].append((client_id, loss_norm))

            augmented_factor = 5  # TODO: avoid hard-coding
            threshold_factor = 1.0
            tuples = []
            already_existing_clients = set()
            for i in range(augmented_factor):
                if start_version - i < 0:
                    break

                tmp = []  # avoid cybil attacks, i.e., outliers repeat deliberately
                for client_id, loss_norm \
                        in self.model_versions_clients_dict[start_version - i]:
                    if client_id in already_existing_clients:
                        continue
                    already_existing_clients.add(client_id)
                    tmp.append((client_id, loss_norm))
                tuples += tmp

            if len(tuples) >= threshold_factor * self.client_per_round:
                logging.info(f"Starting anomaly detection with {len(tuples)} recent records.")
                self.detect_outliers(tuples)
            else:
                logging.info(f"Records collected for anomaly detection are not enough: {len(tuples)}.")

        trainset_size = self.trainset_sizes[client_id]
        norm = loss_norm * trainset_size
        utility = norm
        self.utility[client_id] = utility

        logging.info(f"[Sirius] Client #{client_id}'s utility "
                     f"is updated by loss {round(loss_norm, 2)} to {round(utility, 2)}. "
                     f"Interval: "
                     f"{[round(training_begin_time, 2), round(training_end_time, 2)]}, "
                     f"latency: {round(response_latency, 2)}, trainset_size: {trainset_size}.")

    def staleness_factor_calculator(self, lag, agg_version):
        return 1.0 / pow(lag + 1, self.staleness_penalty_factor)

    def take_random_clients(self, source_list, num_to_take):
        return random.sample(source_list, num_to_take)

    def take_top_clients(self, score_dict, source_list, num_to_take):
        mode = "deterministic"

        raw_sub_score_dict = {
            client_id: score
            for client_id, score in score_dict.items()
            if client_id in source_list
        }

        # break the order for load balancing,
        # utilizing the fact that sorted is stable
        keys = list(raw_sub_score_dict.keys())
        random.shuffle(keys)
        sub_score_dict = {k: raw_sub_score_dict[k] for k in keys}

        sorted_clients = sorted(sub_score_dict,
                                key=sub_score_dict.get,
                                reverse=True)
        if sorted_clients:
            if mode == "probabilistic":
                actual_num_to_take = len(sorted_clients) \
                    if num_to_take > len(sorted_clients) \
                    else num_to_take  # for np.random.choice's use
                total_score = sum([score_dict[client_id]
                                   for client_id in sorted_clients])

                if total_score == 0:  # avoid division by zero
                    taken_clients = sorted_clients[:num_to_take]
                else:
                    taken_clients = list(np.random.choice(
                        sorted_clients,
                        actual_num_to_take,
                        p=[score_dict[client_id] / total_score
                           for client_id in sorted_clients],
                        replace=False
                    ).astype(object))
            else:
                taken_clients = sorted_clients[:num_to_take]
        else:
            taken_clients = []

        logging.info(f"[Sirius] [Debug] For taking top {num_to_take} clients: "
                     f"score_dict: {score_dict}, "
                     f"source_list: {source_list}, "
                     f"sorted_clients: {sorted_clients}, "
                     f"taken_clients: {taken_clients}")
        return taken_clients

    def choose_clients(self, available_clients, num_to_select, use_cache=True):
        if use_cache and self.done_first_selection:
            result = self.clients_to_select
            self.clients_to_select = []
            return result

        if self.robustness:
            outliers = [client_id for client_id in available_clients
                        if client_id in self.detected_corrupted_clients]
            available_clients = [client_id for client_id in available_clients
                                 if client_id not in self.detected_corrupted_clients]
            logging.info(f"These clients are detected as outliers "
                         f"and precluded from selection: {outliers}.")

        if self.client_selection == "random":
            clients_to_select = self.take_random_clients(
                source_list=available_clients, num_to_take=num_to_select)
            logging.info(
                f"[Sirius] [Debug] Random selection result: "
                f"{clients_to_select}.")
        else:
            unexplored_clients = list(
                (set(range(1, self.total_clients + 1)) -
                 set(self.explored_clients)).intersection(set(available_clients)))
            explored_clients = list(
                set(available_clients) -
                set(available_clients).intersection(set(unexplored_clients)))

            # why not use num_to_select * self.explore_factor:
            # unfriendly when num_to_select is small
            planned_explore_len = min(
                len(unexplored_clients),
                np.random.binomial(num_to_select, self.explore_factor, 1)[0])
            self.explore_factor = max(
                self.explore_factor * self.exploration_decaying_factor,
                self.min_explore_factor)

            actual_exploit_len = min(len(explored_clients),
                                     num_to_select - planned_explore_len)
            actual_explore_len = min(len(unexplored_clients),
                                     num_to_select - actual_exploit_len)

            # use last lag to calculate penalty
            score_dict = {}
            debug_dict = {}
            for client_id, client_utility in self.utility.items():
                last_response_latency = self.last_response_latency[client_id]
                debug_dict[client_id] = {}

                if self.independent_selection or self.statistical_only:
                    score = client_utility
                else:
                    speed_penalty_factor = self.speed_penalty_factor \
                                           * (1 - num_to_select / self.client_per_round)
                    speed_penalty = (1. / last_response_latency) ** speed_penalty_factor
                    debug_dict[client_id].update({
                        "resp_lat": round(last_response_latency, 4),
                        "spd_pnt_fac": round(speed_penalty_factor, 4),
                        "spd_pnt": round(speed_penalty, 4),
                    })
                    score = client_utility * speed_penalty

                if self.staleness_discounted:
                    # moving average
                    last_lag_list = self.get_async_client_lag(client_id, multiple=5)  # TODO: avoid hard-coding
                    if last_lag_list:
                        last_lag = np.mean(last_lag_list)
                    else:
                        last_lag = 0

                    staleness_penalty_factor = self.staleness_factor_calculator(last_lag, 0)
                    score *= staleness_penalty_factor
                    debug_dict[client_id].update({
                        "stale_pen_fac": round(staleness_penalty_factor, 4),
                        "last_lag_list": last_lag_list,
                    })

                score_dict[client_id] = score
                debug_dict[client_id].update({
                    "score": round(score, 4),
                    "utility": round(client_utility, 4)
                })
            # logging.info(f"[Sirius] In choose_clients debug_dict: {debug_dict}.")

            clients_to_exploit = self.take_top_clients(
                score_dict=score_dict,
                source_list=explored_clients,
                num_to_take=actual_exploit_len)

            if self.independent_selection:
                score_dict_2 = {}
                for client_id, _ in self.utility.items():
                    last_response_latency = self.last_response_latency[client_id]
                    speed_penalty_factor = self.speed_penalty_factor
                    speed_penalty = (1. / last_response_latency) ** speed_penalty_factor
                    score_dict_2[client_id] = speed_penalty

                clients_to_exploit_2 = self.take_top_clients(
                    score_dict=score_dict_2,
                    source_list=explored_clients,
                    num_to_take=actual_exploit_len)

                assert isinstance(clients_to_exploit_2, list)  # ensure concatenation
                clients_to_exploit_inter = list(set(clients_to_exploit)
                                                .intersection(set(clients_to_exploit_2)))
                clients_to_exploit_not_inter = [e for e in (clients_to_exploit + clients_to_exploit_2)
                                                if e not in clients_to_exploit_inter]
                clients_to_exploit_mixed = clients_to_exploit_inter
                clients_to_exploit_mixed += self.take_random_clients(
                    source_list=clients_to_exploit_not_inter,
                    num_to_take=actual_exploit_len - len(clients_to_exploit_inter)
                )
                logging.info(f"Only data quality: {clients_to_exploit}, "
                             f"only speed: {clients_to_exploit_2}, "
                             f"independent mix: {clients_to_exploit_mixed}.")
                clients_to_exploit = clients_to_exploit_mixed

            clients_to_explore = self.take_random_clients(
                source_list=unexplored_clients, num_to_take=actual_explore_len)
            clients_to_select = clients_to_exploit + clients_to_explore

            logging.info(
                f"[Sirius] [Debug] Exploiting "
                f"{actual_exploit_len} clients: {clients_to_exploit}, exploring "
                f"{actual_explore_len} clients: {clients_to_explore}.")

        if not self.done_first_selection and clients_to_select:
            self.done_first_selection = True
        return clients_to_select

    def whether_to_aggregate(self, available_clients, num_to_select, running_clients,
                             done_clients, current_time, current_update_version):
        if len(done_clients) == 0:
            return False

        # avoid selecting contributors in the same round
        self.clients_to_select = self.choose_clients(
            available_clients=available_clients,
            num_to_select=num_to_select,
            use_cache=False
        )

        # threshold aggregation
        if self.aggregation_threshold:
            pacer_approve = len(done_clients) >= self.aggregation_threshold
            logging.info(f"[Sirius] done_clients: {done_clients}, "
                         f"number of done clients: {len(done_clients)}, "
                         f"aggregation_threshold: {self.aggregation_threshold}, "
                         f"to_aggregate: {pacer_approve}.")
            return pacer_approve

        # bounded staleness
        no_staleness_violation = True
        if self.bounded_staleness:
            for client_id in running_clients:
                lag = self.get_async_client_lag(
                    client_id, current_update_version, running=True)
                if lag >= self.staleness_bound:
                    no_staleness_violation = False
                    logging.info(f"Client #{client_id}'s staleness ({lag}) is at the bound.")
                    break

        # pacer
        MAX = 180  # TODO: avoid hard-coding
        pacer_approve = True
        if len(running_clients) == 0:
            logging.info(f"No client is running.")
        else:
            max_resp_latency = -1.0
            max_client_id = -1
            debug_dict = {}

            for client_id in sorted(running_clients):
                last_response_latency = self.last_response_latency[client_id]
                cur_staleness = self.get_async_client_lag(
                    client_id=client_id, current_version=current_update_version, running=True
                )

                first_run = False
                if last_response_latency < 1.0 + EPSILON:  # it has not been involved yet
                    last_response_latency = MAX
                    first_run = True

                if last_response_latency > max_resp_latency:
                    max_client_id = client_id
                    max_resp_latency = last_response_latency
                debug_dict[client_id] = {
                    'first_run': first_run,
                    'last_resp': last_response_latency,
                    'cur_stale': cur_staleness,
                }

            agg_interval = max_resp_latency / self.staleness_bound  # TODO: avoid hard-coding
            last_agg_time = self.start_time
            if len(self.agg_time) > 1:  # at least one model aggregation is done
                last_agg_time = self.agg_time[-1][0]
            elasped_time = current_time - last_agg_time
            pacer_approve = elasped_time >= agg_interval

            logging.info(f"Running clients: {debug_dict}, max_resp_latency: {max_resp_latency}, "
                         f"max_client_id: {max_client_id}, staleness_bound: {self.staleness_bound}, "
                         f"agg_interval: {agg_interval}, elasped_time: {elasped_time}, "
                         f"current_version: {current_update_version}.")

        to_aggregate = pacer_approve and no_staleness_violation
        logging.info(
            f"no_staleness_violation: {no_staleness_violation}, "
            f"pacer_approve: {pacer_approve}.")

        return to_aggregate

    def has_done_first_selection(self):
        return self.done_first_selection
