"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""
import logging
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from plato.config import Config

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.datasource = datasource
        self.sample_weights = None
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0
        self.update_concentration(concentration)

        partition_size = Config().data.partition_size
        self.update_partition_size(partition_size)

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        return WeightedRandomSampler(weights=self.sample_weights,
                                     num_samples=self.partition_size,
                                     replacement=False,
                                     generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return self.partition_size

    def update_partition_size(self, partition_size):
        self.partition_size = partition_size

    def update_concentration(self, concentration):
        # The list of labels (targets) for all the examples
        target_list = self.datasource.targets()
        class_list = self.datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list)))

        logging.info("[Client #%d] [Dirichlet] Target proportions: %s.",
                     self.client_id, target_proportions)

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[random.randint(0, len(class_list) - 1)] = 1

        self.sample_weights = target_proportions[target_list]