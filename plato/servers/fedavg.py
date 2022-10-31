"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os
import random
import time
import numpy as np
import torch
from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor

from plato.servers import base as base


class Server(base.Server):
    """Federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__()

        if hasattr(Config().trainer, 'use_wandb'):
            import wandb

            wandb.init(project="plato", reinit=True)

        self.model = model
        self.algorithm = algorithm
        self.trainer = trainer

        self.testset = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round
        self.waiting_to_start = False
        recorded_items = Config().results.types

        if hasattr(Config().server, 'seconds'):
            self.seconds = Config().server.seconds
        else:
            self.seconds = None

        if hasattr(Config().server, 'asynchronous'):
            self.current_step = 0
            self.current_update_version = 0
            self.total_steps = Config().server.seconds \
                               // Config().server.asynchronous.seconds_per_step

            self.available_clients = {}
            self.selected_clients = set()

            if hasattr(Config(), 'results'):
                self.recorded_items = ['step'] + [
                    x.strip() for x in recorded_items.split(',')
                ]
            if hasattr(Config().server.asynchronous, 'num_running') \
                    and Config().server.asynchronous.num_running == 'limited' \
                    or hasattr(Config().server.asynchronous, 'sirius'):
                logging.info(
                    "[Server #%d] Started training on %s clients with %s "
                    "in total at any time.", os.getpid(), self.total_clients,
                    self.clients_per_round)
            else:  # in other words, default is unlimited
                logging.info(
                    "[Server #%d] Started training on %s clients with %s per step.",
                    os.getpid(), self.total_clients, self.clients_per_round)
        else:
            if hasattr(Config(), 'results'):
                recorded_items = Config().results.types
                self.recorded_items = ['round'] + [
                    x.strip() for x in recorded_items.split(',')
                ]
            logging.info(
                "[Server #%d] Started training on %s clients with %s per round.",
                os.getpid(), self.total_clients, self.clients_per_round)

        # starting time of a global training round
        self.start_time = time.perf_counter()
        self.client_manager.set_global_start_timestamp(self.start_time)

        random.seed()

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """

        logging.info("[Server #%d] Configuring the server...", os.getpid())

        if hasattr(Config().trainer, 'target_accuracy'):
            target_accuracy = Config().trainer.target_accuracy
        else:
            target_accuracy = None

        if hasattr(Config().server, 'asynchronous'):
            if target_accuracy:
                logging.info("Training: %s seconds or %s%% accuracy\n",
                             self.seconds, 100 * target_accuracy)
            else:
                logging.info("Training: %s seconds\n", self.seconds)
        else:
            if self.seconds:
                if target_accuracy:
                    logging.info("Training: %s seconds or %s%% accuracy\n",
                                 self.seconds, 100 * target_accuracy)
                else:
                    logging.info("Training: %s seconds\n", self.seconds)
            else:
                total_rounds = Config().trainer.rounds
                if target_accuracy:
                    logging.info("Training: %s rounds or %s%% accuracy\n",
                                 total_rounds, 100 * target_accuracy)
                else:
                    logging.info("Training: %s rounds\n", total_rounds)

        self.load_trainer()

        if not Config().clients.do_test:
            dataset = datasources_registry.get(client_id=0)
            self.testset = dataset.get_test_set()

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    def load_trainer(self):
        """Setting up the global model to be trained via federated learning."""
        if self.trainer is None:
            self.trainer = trainers_registry.get(model=self.model)

        self.trainer.set_client_id(0)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

    async def register_client(self, sid, data):
        """Adding a newly arrived client to the list of clients."""
        client_id = data['id']
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            if hasattr(Config().server, 'asynchronous'):
                self.available_clients[client_id] = {
                    'sid': sid,
                    'available_from': time.perf_counter()
                }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)

            trainset_size = data['trainset_size']
            # should not be used with customized latency
            if hasattr(Config().data, 'partition_size_update') and not \
                    (hasattr(Config().server, 'response_latency_distribution')
                     and Config().server.response_latency_distribution.name == "customized"):
                partition_size = self.client_manager \
                    .get_updated_partition_size(client_id)
                await self.sio.emit('update_partition_size',
                                    {
                                        'id': client_id,
                                        'partition_size': partition_size
                                    }, room=sid)

                # This is the new trainset_size
                trainset_size = partition_size
            self.client_manager.set_trainset_size(client_id, trainset_size)

            if hasattr(Config().server, 'client_selection') \
                    and Config().server.client_selection.name == 'oort':
                feedbacks = {
                    'reward': trainset_size,
                    'duration': 1,
                }  # duration will be updated after participation
                self.client_manager.init_client_selector(name="oort",
                                                         client_id=client_id,
                                                         feedbacks=feedbacks)

            if hasattr(Config().server, 'response_latency_distribution') \
                    and Config().server.response_latency_distribution.name \
                    == "customized":
                self.client_manager.update_resp_lat(
                    name="customized",
                    client_id=client_id,
                    trainset_size=data['trainset_size'],
                )

            # only make sense when sampler is noniid (dirichlet)
            if hasattr(Config().data, 'concentration_update'):
                concentration = self.client_manager\
                    .get_updated_concentration(client_id)
                await self.sio.emit('update_concentration',
                                    {
                                        'id': client_id,
                                        'concentration':
                                            concentration.astype(object)
                                    }, room=sid)

            # only make sense when we need to simulate data corruption
            if hasattr(Config().data, 'data_corruption'):
                corrupted_clients = self.client_manager\
                    .get_corruption_simulation_plan()
                if client_id in corrupted_clients:
                    await self.sio.emit('do_data_corruption',
                                        {
                                            'id': client_id,
                                        }, room=sid)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

            if client_id in self.selected_clients \
                    and client_id not in self.payload_done_clients:
                await self.sio.emit('resend_request', {'id': client_id},
                                    room=sid)
                logging.info(
                    f"[Server #%d] Continue waiting for Client #%d's "
                    f"payload", os.getpid(), client_id)

        if not hasattr(Config().server, 'asynchronous') \
                and self.current_round == 0 and not self.waiting_to_start:

            # first yield to others for allowing more clients to register
            self.waiting_to_start = True
            await asyncio.sleep(5)

            if hasattr(Config().server, 'overselection'):
                overselection_factor = Config().server.overselection
                actual_clients_per_round = int(np.ceil(
                    overselection_factor * self.clients_per_round))
                if len(self.clients) < actual_clients_per_round:
                    return
            elif len(self.clients) < self.clients_per_round:
                return

            logging.info("[Server #%d] Starting training.", os.getpid())
            await self.select_clients()

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        if not hasattr(Config().server, "asynchronous") \
                and hasattr(Config().server, "overselection"):
            overselection_factor = Config().server.overselection
            intended_sample_size = int(np.ceil(
                self.clients_per_round * overselection_factor))
            actual_clients_pool = list(set(self.clients_pool)
                                       - self.clients_to_discard)
        elif hasattr(Config().server, "asynchronous") \
                and hasattr(Config().server.asynchronous, "fedbuff"):
            # selected_clients: running clients, bounded maximum concurrency
            intended_sample_size = self.clients_per_round - len(self.selected_clients)
            actual_clients_pool = self.clients_pool
        else:
            intended_sample_size = self.clients_per_round
            actual_clients_pool = self.clients_pool

        sample_size = min(intended_sample_size, len(actual_clients_pool))
        logging.info("[Server #%d] To select %d from client pool: %s.",
                     os.getpid(), sample_size, actual_clients_pool)
        return self.client_manager.choose_clients(actual_clients_pool,
                                                  sample_size)

    def extract_client_updates(self, updates, custom_baseline_weights=None):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, __, payload) in updates]
        return self.algorithm.compute_weight_updates(weights_received,
                                                     custom_baseline_weights)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        factors = [1.0] * len(updates)
        if hasattr(Config().server, 'asynchronous') \
                and (hasattr(Config().server.asynchronous, "staleness_aware_lr")
                     and Config().server.asynchronous.staleness_aware_lr == "reciprocal"
                     or hasattr(Config().server.asynchronous, 'sirius')
                     or hasattr(Config().server.asynchronous, 'fedbuff')
                    ):

            self.client_manager.record_aggregation(
                client_id_list=self.payload_done_clients,
                agg_time=time.perf_counter(),
                model_version=self.current_update_version
            )

            # note that self.payload_done_clients are in the same order as self.updates
            for i, client_id in enumerate(self.payload_done_clients):
                lag = self.client_manager.get_async_client_lag(client_id=client_id)
                factors[i] = self.client_manager.staleness_factor_calculator(
                    lag=lag, agg_version=self.current_update_version)

        for i, update in enumerate(weights_received):
            __, report, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples /
                                             self.total_samples) * factors[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        model_name = Config().trainer.model_name

        while self.parent_pipe.poll():
            begin_time, self.accuracy = self.parent_pipe.recv()
            if hasattr(Config().server, 'asynchronous') \
                    and hasattr(Config().server.asynchronous, 'sirius'):

                pseudo_acc = self.accuracy
                if 'albert' in model_name:
                    maximum_perplexity = 40000 # avoid hard-coding
                    pseudo_acc = (maximum_perplexity - self.accuracy) / maximum_perplexity

        if hasattr(Config(), 'results'):
            temp_dict = {}
            if hasattr(Config().server, 'asynchronous'):
                temp_dict.update({
                    'step': self.current_step,
                    'elapsed_time': time.perf_counter() - self.start_time
                })
            else:
                temp_dict.update({
                    'round':
                    self.current_round,
                    'training_time':
                    max([
                        report.training_time
                        for (__, report, __) in self.updates
                    ]),
                    'round_time':
                    time.perf_counter() - self.start_time
                })

            if "albert" in model_name:  # actually perplexity
                temp_dict.update({
                    'perplexity': self.accuracy,
                })
            else:
                temp_dict.update({
                    'accuracy': self.accuracy * 100,
                })

            new_row = []
            for item in self.recorded_items:
                item_value = temp_dict[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

    @staticmethod
    def accuracy_averaging(reports):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum([report.num_samples for (__, report, __) in reports])

        # Perform weighted averaging
        accuracy = 0
        for (__, report, __) in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """
        return payload
