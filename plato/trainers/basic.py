"""
The training and testing loops for PyTorch.
"""
import copy
import logging
import multiprocessing as mp
import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import base
from plato.utils import optimizers
import pickle


def customized_save(folder, filename, obj):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as fout:
        pickle.dump(obj, fout)


def customized_load(folder, filename):
    file_path = os.path.join(folder, filename)
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)


class Trainer(base.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__()

        if model is None:
            model = models_registry.get()

        # Use data parallelism if multiple GPUs are available and the configuration specifies it
        if Config().is_parallel():
            logging.info("Using Data Parallelism.")
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        if hasattr(Config().server, 'client_selection') \
                and Config().server.client_selection.name == 'oort' or \
                hasattr(Config().server, 'asynchronous') and \
                hasattr(Config().server.asynchronous, 'sirius'):
            self.moving_loss_norm = 0.  # initial value

        if hasattr(Config().clients, 'async_training') \
                and Config().clients.async_training is True:
            self.train_proc = None
            self.tic = None
            self.async_training_begun = False

    def zeros(self, shape):
        """Returns a PyTorch zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None):
        """Saving the model to a file."""
        model_name = Config().trainer.model_name
        model_dir = Config().result_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.pth'

        torch.save(self.model.state_dict(), model_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id,
                         model_path)

    def load_model(self, filename=None):
        """Loading pre-trained model weights from a file."""
        model_dir = Config().result_dir
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s.",
                         self.client_id, model_path)

        self.model.load_state_dict(torch.load(model_path))

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                _train_loader = getattr(self, "train_loader", None)

                if callable(_train_loader):
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=sampler.get())

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                # Initializing the optimizer
                get_optimizer = getattr(self, "get_optimizer",
                                        optimizers.get_optimizer)
                optimizer = get_optimizer(self.model)

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None

                pause_interval = 5
                yield_time = 1
                train_start_time = time.perf_counter()

                if hasattr(Config().server, 'client_selection') \
                        and Config().server.client_selection.name == 'oort' \
                        or hasattr(Config().server, 'asynchronous') \
                        and hasattr(Config().server.asynchronous, 'sirius'):
                    epoch_train_loss = 1e-4
                    loss_decay = 1e-2  # TODO: avoid magic numbers

                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(train_loader):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        optimizer.zero_grad()

                        model_name = Config().trainer.model_name
                        if 'albert' in model_name:
                            outputs = self.model(examples, labels=labels)
                            loss = outputs[0]
                        else:
                            if cut_layer is None:
                                outputs = self.model(examples)
                            else:
                                outputs = self.model.forward_from(
                                    examples, cut_layer)
                            loss = loss_criterion(outputs, labels)

                        # collect training feedback according to Oort's code
                        if (hasattr(Config().server, 'client_selection')
                            and Config().server.client_selection.name == 'oort'
                            or hasattr(Config().server, 'asynchronous')
                            and hasattr(Config().server.asynchronous, 'sirius')) \
                                and epoch == 1:
                            loss_list = loss.tolist()
                            if isinstance(loss_list, list):
                                temp_loss = sum([l**2 for l in loss_list
                                                 ]) / float(len(loss_list))
                            else:
                                temp_loss = loss.data.item()
                                if 'albert' in model_name:
                                    temp_loss /= len(labels)

                            if epoch_train_loss == 1e-4:
                                epoch_train_loss = temp_loss
                            else:
                                epoch_train_loss = (1. - loss_decay) * epoch_train_loss \
                                                   + loss_decay * temp_loss

                        loss.backward()

                        optimizer.step()

                        if lr_schedule is not None:
                            lr_schedule.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))

                        train_latency = time.perf_counter() - train_start_time
                        if train_latency > pause_interval:
                            time.sleep(yield_time)
                            train_start_time = time.perf_counter()

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if hasattr(Config().server, 'client_selection') \
                and Config().server.client_selection.name == 'oort' or \
                hasattr(Config().server, 'asynchronous') and \
                hasattr(Config().server.asynchronous, 'sirius'):
            self.moving_loss_norm = np.sqrt(epoch_train_loss)

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

            if hasattr(Config().server, 'client_selection') \
                    and Config().server.client_selection.name == 'oort' or \
                    hasattr(Config().server, 'asynchronous') and \
                    hasattr(Config().server.asynchronous, 'sirius'):
                customized_save(folder='./customized_save/',
                                filename=f"{self.client_id}_util.pkl",
                                obj=self.moving_loss_norm)

        if 'use_wandb' in config:
            run.finish()

    def train(self, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.

        Returns:
        bool: Whether training was successfully completed.
        float: Elapsed time during training.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        max_concurrency_specified = hasattr(Config().trainer,
                                            'max_concurrency')
        is_async_training = hasattr(Config().clients, 'async_training') and \
                            Config().clients.async_training is True

        if max_concurrency_specified or is_async_training:

            if (is_async_training and not self.async_training_begun) \
                    or (not is_async_training and max_concurrency_specified):
                self.start_training()

                if is_async_training:
                    self.tic = time.perf_counter()
                else:
                    tic = time.perf_counter()

                if mp.get_start_method(allow_none=True) != 'spawn':
                    mp.set_start_method('spawn', force=True)

                train_proc = mp.Process(target=self.train_process,
                                        args=(
                                            config,
                                            trainset,
                                            sampler,
                                            cut_layer,
                                        ))
                train_proc.start()

                if is_async_training:
                    # to avoid the case when is_alive() does not become
                    # True immediately after calling start()
                    while not train_proc.is_alive():
                        time.sleep(0.1)
                    self.async_training_begun = True
                    self.train_proc = train_proc
                    return

            if (is_async_training and self.async_training_begun) \
                    or (not is_async_training and max_concurrency_specified):

                if is_async_training:
                    train_proc = self.train_proc
                    tic = self.tic
                    self.async_training_begun = False
                else:
                    train_proc.join()
                logging.info(
                    f"Client #{self.client_id}'s exit code: {train_proc.exitcode}"
                )

                model_name = Config().trainer.model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

                try:
                    self.load_model(filename)
                except OSError as error:  # the model file is not found, training failed
                    if hasattr(Config().trainer, 'max_concurrency'):
                        self.run_sql_statement(
                            "DELETE FROM trainers WHERE run_id = (?)",
                            (self.client_id, ))
                    raise ValueError(
                        f"Training on client {self.client_id} failed ({error})."
                    ) from error

                if hasattr(Config().server, 'client_selection') \
                        and Config().server.client_selection.name == 'oort' or \
                        hasattr(Config().server, 'asynchronous') and \
                        hasattr(Config().server.asynchronous, 'sirius'):
                    self.moving_loss_norm = customized_load(
                        folder='./customized_save/',
                        filename=f"{self.client_id}_util.pkl")

                toc = time.perf_counter()
                self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, cut_layer)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    def get_utility(self):
        return self.moving_loss_norm

    def async_training_ended(self) -> bool:
        return self.train_proc and not self.train_proc.is_alive()

    def reset_async_training_status(self) -> None:
        self.train_proc = None

    def test_process(self, config, testset):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        """
        self.model.to(self.device)
        self.model.eval()

        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=config['batch_size'], shuffle=False)

                correct = 0
                total = 0
                overall_loss = 0

                with torch.no_grad():
                    for examples, labels in test_loader:
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        model_name = Config().trainer.model_name
                        if 'albert' in model_name:
                            outputs = self.model(examples, labels=labels)
                            loss_value = outputs[0].data.item()
                            overall_loss += loss_value
                        else:
                            outputs = self.model(examples)

                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                if 'albert' in model_name:  # actually it is perplexity
                    overall_loss /= len(test_loader)
                    accuracy = np.exp(overall_loss)
                else:
                    accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def test(self, testset) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            self.start_training()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.test_process,
                              args=(
                                  config,
                                  testset,
                              ))
            proc.start()
            proc.join()

            try:
                model_name = Config().trainer.model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Testing on client #{self.client_id} failed.") from error

            self.pause_training()
        else:
            accuracy = self.test_process(config, testset)

        return accuracy

    def server_test(self, testset, mode="sync"):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        # in case that model is changed by aggregated
        model_to_test = None
        if mode == "async":
            model_to_test = copy.deepcopy(self.model)
        else:
            model_to_test = self.model

        model_to_test.to(self.device)
        model_to_test.eval()

        custom_test = getattr(self, "test_model", None)

        if callable(custom_test):
            return self.test_model(config, testset)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False)

        correct = 0
        total = 0
        overall_loss = 0.0

        model_name = Config().trainer.model_name
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                if 'albert' in model_name:  # actually perplexity
                    outputs = model_to_test(examples, labels=labels)
                    loss_value = outputs[0].data.item()
                    overall_loss += loss_value
                else:
                    outputs = model_to_test(examples)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        if mode == "async":
            del model_to_test

        if 'albert' in model_name:  # actually it is perplexity
            overall_loss /= len(test_loader)
            accuracy = np.exp(overall_loss)
        else:
            accuracy = correct / total

        gc.collect()
        return accuracy
