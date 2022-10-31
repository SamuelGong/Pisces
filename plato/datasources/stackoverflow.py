"""
The StackOverflow dataset.
"""

import pickle
import logging
import os
import torch
from torch.utils.data import Dataset

from plato.config import Config
from plato.datasources import base


class CustomTextDataset(Dataset):
    def __init__(self, loaded_data):
        super().__init__()
        self.inputs = loaded_data['x']
        self.labels = loaded_data['y']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.long)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, labels


class DataSource(base.DataSource):
    def __init__(self, client_id=0):
        super().__init__()
        self.trainset = None
        self.testset = None

        root_path = os.path.join(Config().data.data_path, 'Reddit',
                                 'packaged_data')
        if client_id == 0:
            # If we are on the federated learning server
            data_dir = os.path.join(root_path, 'test')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/Reddit/test/" \
                       + str(client_id) + ".zip"
        else:
            data_dir = os.path.join(root_path, 'train')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/Reddit/train/" \
                       + str(client_id) + ".zip"

        if not os.path.exists(os.path.join(data_dir, str(client_id))):
            logging.info(
                "Downloading the Reddit dataset "
                "with the client datasets pre-partitioned. This may take a while.",
            )
            self.download(url=data_url, data_path=data_dir)

        loaded_data = DataSource.read_data(
            file_path=os.path.join(data_dir, str(client_id)))
        dataset = CustomTextDataset(loaded_data=loaded_data)

        if client_id == 0:  # testing dataset on the server
            self.testset = dataset
        else:  # training dataset on one of the clients
            self.trainset = dataset

    @staticmethod
    def read_data(file_path):
        """ Reading the dataset specific to a client_id. """
        with open(file_path, 'rb') as fin:
            loaded_data = pickle.load(fin)
        return loaded_data

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)