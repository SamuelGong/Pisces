import logging
import numpy as np
import random
from plato.client_managers import base
from plato.config import Config


class ClientManager(base.ClientManager):
    def __init__(self):
        super().__init__()
        self.client_per_round = Config().clients.per_round

        params = Config().server.asynchronous.fedbuff
        self.staleness_penalty_factor = params.staleness_penalty_factor
        self.threshold_aggregation = params.threshold_aggregation

        self.aggregation_threshold = max(1, int(np.floor(
            params.threshold_aggregation * self.client_per_round
        )))

        self.seed = params.seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def whether_to_aggregate(self, num_done_clients):
        return num_done_clients >= self.aggregation_threshold

    def staleness_factor_calculator(self, lag, agg_version):
        return 1.0 / pow(lag + 1, self.staleness_penalty_factor)