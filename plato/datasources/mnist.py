"""
The MNIST dataset from the torchvision package.
"""
import logging
from torchvision import datasets, transforms
import torch
import numpy as np
from plato.config import Config
from plato.datasources import base


class CorruptedMNIST(torch.utils.data.Dataset):
    def __init__(self, orig, seed):
        super(CorruptedMNIST, self).__init__()
        self.orig_mnist = orig
        np.random.seed(seed)
        self.num_classes = 10

    def __getitem__(self, index):
        x, y = self.orig_mnist[index]  # get the original item
        my_x = x
        if hasattr(Config().data, "data_corruption") \
                and Config().data.data_corruption.type == "flip":
            my_y = np.random.choice(np.arange(0,
                                              self.num_classes), 1)[0].item()
        else:  # it should not get here
            assert 0
        return my_x, my_y

    def __len__(self):
        return self.orig_mnist.__len__()


class DataSource(base.DataSource):
    """ The MNIST dataset. """
    def __init__(self):
        super().__init__()
        self._path = Config().data.data_path

        self.basic_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]

        _transform = transforms.Compose(self.basic_transform)
        self.trainset = datasets.MNIST(root=self._path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=self._path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000

    def simulate_data_corruption(self, client_id):
        if hasattr(Config().data, "data_corruption") \
                and Config().data.data_corruption.type == "flip":
            seed = Config().data.data_corruption.seed
            client_seed = seed * client_id

            self.trainset = CorruptedMNIST(self.trainset, client_seed)
            logging.info(f"[Client #{client_id}] Data corrupted "
                         f"with seed {client_seed}.")
            return self.trainset
