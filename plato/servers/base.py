"""
The base class for federated learning servers.
"""
import copy
import gc

import numpy as np
import torch
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from abc import abstractmethod

import asyncio
import socketio
from aiohttp import web
import aiohttp
from distutils.version import StrictVersion
from plato.client import run
from plato.config import Config
from plato.utils import s3
from plato.client_managers import registry as client_managers_registry


def calc_sleep_time(sec_per_step, cur_step, start_time, gap=0):
    expected_time = sec_per_step * cur_step
    actual_time = time.perf_counter() - start_time
    start_time_drift = actual_time - expected_time
    sleep_time = max(gap, sec_per_step - start_time_drift)
    return sleep_time


def testing_process(server):
    test_interval = Config().server.test_interval_in_seconds

    time.sleep(30)  # reserve some time for the training logic to warm up
    overall_start_time = time.perf_counter()
    testing_times = 0
    while True:
        if server.child_pipe.poll():
            message = server.child_pipe.recv()
            if message == 'exit':
                break

        logging.info("[Server #%d] Testing started.", os.getpid())
        begin_time = time.perf_counter()
        server.process_aggregated_weights()
        server.child_pipe.send((begin_time, server.accuracy))
        logging.info("[Server #%d] Testing ended.", os.getpid())

        sleep_time = calc_sleep_time(test_interval, testing_times, overall_start_time)
        time.sleep(sleep_time)
        testing_times += 1

    logging.info(
        f"[Server #%d] The testing process exits "
        f"as the training one exited.", os.getpid())
    server.child_pipe.close()
    sys.exit(0)


class ServerEvents(socketio.AsyncNamespace):
    """ A custom namespace for socketio.AsyncServer. """
    def __init__(self, namespace, plato_server):
        super().__init__(namespace)
        self.plato_server = plato_server

    #pylint: disable=unused-argument
    async def on_connect(self, sid, environ):
        """ Upon a new connection from a client. """
        logging.info("[Server #%d] A new client just connected.", os.getpid())

    async def on_disconnect(self, sid):
        """ Upon a disconnection event. """
        logging.info("[Server #%d] An existing client just disconnected.",
                     os.getpid())
        await self.plato_server.client_disconnected(sid)

    async def on_client_alive(self, sid, data):
        """ A new client arrived or an existing client sends a heartbeat. """
        await self.plato_server.register_client(sid, data)

    async def on_client_report(self, sid, data):
        """ An existing client sends a new report from local training. """
        await self.plato_server.client_report_arrived(sid, data['report'])

    async def on_chunk(self, sid, data):
        """ A chunk of data from the server arrived. """
        client_id = None
        for c, d in dict(self.plato_server.clients).items():
            if d['sid'] == sid:
                client_id = c
                break
        if client_id is None:
            pass

        await self.plato_server.client_chunk_arrived(sid, data['data'])

    async def on_client_payload(self, sid, data):
        """ An existing client sends a new payload from local training. """
        await self.plato_server.client_payload_arrived(sid, data['id'])

    async def on_client_payload_done(self, sid, data):
        """ An existing client finished sending its payloads from local training. """
        await self.plato_server.client_payload_done(sid, data['id'],
                                                    data['obkey'])


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.sio = None
        self.client = None
        self.clients = {}
        self.total_clients = 0
        # The client ids are stored for client selection
        self.clients_pool = []
        self.clients_per_round = 0
        self.selected_clients = []
        self.current_round = 0
        self.algorithm = None
        self.trainer = None
        self.accuracy = 0
        self.reports = {}
        self.updates = []
        self.client_payload = {}
        self.client_chunks = {}
        self.s3_client = None
        self.do_want_to_disconnect = []
        self.payload_done_clients = []
        self.parent_pipe = None
        self.child_pipe = None
        self.client_manager = client_managers_registry.get()

        if hasattr(Config().server, 'overselection'):
            self.clients_to_discard = set()

    def run(self,
            client=None,
            edge_server=None,
            edge_client=None,
            trainer=None):
        """Start a run loop for the server. """
        # Remove the running trainers table from previous runs.
        if not Config().is_edge_server() and hasattr(Config().trainer,
                                                     'max_concurrency'):
            with Config().sql_connection:
                Config().cursor.execute("DROP TABLE IF EXISTS trainers")

        self.client = client
        self.configure()

        if Config().is_central_server():
            # In cross-silo FL, the central server lets edge servers start first
            # Then starts their clients
            Server.start_clients(as_server=True,
                                 client=self.client,
                                 edge_server=edge_server,
                                 edge_client=edge_client,
                                 trainer=trainer)

            # Allowing some time for the edge servers to start
            time.sleep(5)

        if hasattr(Config().server,
                   'disable_clients') and Config().server.disable_clients:
            logging.info(
                "No clients are launched (server:disable_clients = true)")
        else:
            Server.start_clients(client=self.client)

        self.start()

    async def aggregate_and_reset_for_async(self):
        if len(self.updates) > 0:
            client_list = sorted(self.payload_done_clients)
            current_update_version = self.current_update_version
            staleness_list = [self.client_manager.get_async_client_lag(
                client_id, current_update_version, running=False) for client_id in client_list]
            logging.info(
                "[Server #%d] Received %d clients' reports: "
                "%s with staleness %s, Processing.",
                os.getpid(), len(self.updates),
                client_list, staleness_list
            )

            # TODO: what if self.updates changes during aggregation
            await self.aggregate_weights(self.updates)
            self.payload_done_clients = []
            self.updates = []
            self.current_update_version += 1
            gc.collect()

    async def async_training_coro(self):
        # loop over steps
        seconds_per_step = Config().server.asynchronous.seconds_per_step
        while True:
            logging.info(
                "[Server #%d] Step %s/%s (%s s) starts. Do periodic tasks.",
                os.getpid(), self.current_step, self.total_steps,
                round(time.perf_counter() - self.start_time, 2))

            # first aggregate updates
            to_aggregate = True
            done_clients = self.payload_done_clients
            available_clients = list(self.available_clients)
            # avoid lapping
            available_clients = [c for c in available_clients
                                 if c not in done_clients]
            selected_clients = self.selected_clients

            if hasattr(Config().server.asynchronous, 'staleness_bound'):
                staleness_bound = Config().server.asynchronous.staleness_bound
                delayed_clients = self.client_manager.get_async_delayed_clients(
                    current_version=self.current_update_version,
                    selected_clients=self.selected_clients,
                    done_clients=self.payload_done_clients)
                if delayed_clients:
                    logging.info(
                        "[Server #%d] The delay of %s exceeds the staleness bound %d. "
                        "Have to wait for them.", os.getpid(), delayed_clients,
                        staleness_bound)
                    to_aggregate = False
            elif hasattr(Config().server.asynchronous, 'sirius'):
                to_aggregate = self.client_manager.whether_to_aggregate(
                    num_to_select=self.clients_per_round,
                    available_clients=available_clients,
                    running_clients=selected_clients,
                    done_clients=done_clients,
                    current_time=time.perf_counter(),
                    current_update_version=self.current_update_version)
            elif hasattr(Config().server.asynchronous, "fedbuff"):
                to_aggregate = self.client_manager.whether_to_aggregate(
                    num_done_clients=len(done_clients)
                )

            logging.info(f"[Server #%d] to_aggregate: %s.",
                         os.getpid(), to_aggregate)
            if to_aggregate:
                await self.aggregate_and_reset_for_async()

            await self.wrap_up_processing_reports()
            await self.wrap_up()

            # then select clients
            to_select = True
            logging.info(f"[Server #%d] to_select: %s.",
                         os.getpid(), to_select)
            if to_select:
                logging.info("[Server #%d] Starting training.", os.getpid())
                await self.select_clients()

            logging.info("[Server #%d] Periodic tasks done.", os.getpid())
            sleep_time = calc_sleep_time(seconds_per_step, self.current_step,
                                         self.start_time)
            await asyncio.sleep(sleep_time)

            logging.info("[Server #%d] Step %s/%s ends.", os.getpid(),
                         self.current_step, self.total_steps)
            self.current_step += 1

    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        logging.info("Starting a server at address %s and port %s.",
                     Config().server.address, port)

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        ping_timeout = Config().server.ping_timeout if hasattr(
            Config().server, 'ping_timeout') else 360
        self.sio = socketio.AsyncServer(async_handlers=True,
                                        ping_interval=ping_interval,
                                        max_http_buffer_size=2**31,
                                        ping_timeout=ping_timeout)
        self.sio.register_namespace(
            ServerEvents(namespace='/', plato_server=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        self.parent_pipe, self.child_pipe = mp.Pipe()
        p = mp.Process(target=testing_process, args=(self, ))
        p.start()

        app = web.Application()
        self.sio.attach(app)

        if hasattr(Config().server, 'asynchronous'):
            current_loop = asyncio.get_event_loop()
            current_loop.create_task(self.async_training_coro())
            # see https://github.com/aio-libs/aiohttp/pull/5572 for details
            if StrictVersion(aiohttp.__version__) < StrictVersion('3.8.0'):
                web.run_app(app, host=Config().server.address, port=port)
            else:
                web.run_app(app,
                            host=Config().server.address,
                            port=port,
                            loop=current_loop)
        else:
            web.run_app(app, host=Config().server.address, port=port)

    @staticmethod
    def start_clients(client=None,
                      as_server=False,
                      edge_server=None,
                      edge_client=None,
                      trainer=None):
        """Starting all the clients as separate processes."""
        starting_id = 1

        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            # In the client simulation mode, we only need to launch a limited
            # number of client objects (same as the number of clients per round)
            client_processes = Config().clients.per_round
        else:
            client_processes = Config().clients.total_clients

        if as_server:
            total_processes = Config().algorithm.total_silos
            starting_id += client_processes
        else:
            total_processes = client_processes

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        for client_id in range(starting_id, total_processes + starting_id):
            if as_server:
                port = int(Config().server.port) + client_id
                logging.info(
                    "Starting client #%d as an edge server on port %s.",
                    client_id, port)
                proc = mp.Process(target=run,
                                  args=(client_id, port, client, edge_server,
                                        edge_client, trainer))
                proc.start()
            else:
                logging.info("Starting client #%d's process.", client_id)
                proc = mp.Process(target=run,
                                  args=(client_id, None, client, None, None,
                                        None))
                proc.start()

    async def close_connections(self):
        """Closing all socket.io connections after training completes."""
        for client_id, client in dict(self.clients).items():
            logging.info("Closing the connection to client #%d.", client_id)
            await self.sio.emit('shutdown', room=client['sid'])

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        if hasattr(Config().server, 'asynchronous'):
            logging.info("[Server #%d] Start to select clients at Step %s.",
                         os.getpid(), self.current_step)

            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                raise NotImplementedError

            self.clients_pool = list(self.available_clients)  # extracts keys
        else:
            self.current_round += 1  # only useful in synchronous training
            self.payload_done_clients = []
            self.updates = []
            logging.info("[Server #%d] Starting round %s (%s s).", os.getpid(),
                         self.current_round,
                         round(time.perf_counter() - self.start_time, 2))

            if hasattr(Config().clients, 'simulation') and Config(
            ).clients.simulation and not Config().is_central_server:
                # In the client simulation mode, the client pool for client selection contains
                # all the virtual clients to be simulated
                self.clients_pool = list(range(1, 1 + self.total_clients))

            else:
                # If no clients are simulated, the client pool for client selection consists of
                # the current set of clients that have contacted the server
                self.clients_pool = list(self.clients)

        newly_selected_clients = self.choose_clients()  # that uses self.clients_pool
        logging.info("[Server #%d] Newly selected: %s.",
                     os.getpid(), newly_selected_clients)
        model_version = None

        if hasattr(Config().server, 'asynchronous'):
            model_version = self.current_update_version
            for client_id in newly_selected_clients:
                self.selected_clients.add(client_id)
        else:
            if hasattr(Config().server, 'overselection'):
                self.selected_clients += newly_selected_clients
            else:
                self.selected_clients = newly_selected_clients

        for client_id in newly_selected_clients:
            self.client_manager.record_training_start(
                client_id=client_id,
                begin_time=time.perf_counter(),
                model_version=model_version
            )

        if len(newly_selected_clients) > 0:
            for i, selected_client_id in enumerate(newly_selected_clients):
                if hasattr(Config().clients, 'simulation') \
                        and Config().clients.simulation and not Config().is_central_server:
                    client_id = i + 1
                else:
                    client_id = selected_client_id

                if hasattr(Config().server, 'asynchronous'):
                    del self.available_clients[client_id]

                sid = self.clients[client_id]['sid']

                logging.info("[Server #%d] Selecting client #%d for training.",
                             os.getpid(), selected_client_id)

                server_response = {'id': selected_client_id}
                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                # Sending the server payload to the client
                logging.info(
                    "[Server #%d] Sending the current model to client #%d.",
                    os.getpid(), selected_client_id)
                await self.send(sid, payload, selected_client_id)

                if hasattr(Config().server, 'asynchronous') \
                        and (hasattr(Config().server.asynchronous, 'num_running')
                             and Config().server.asynchronous.num_running == 'limited'
                             or hasattr(Config().server.asynchronous, 'sirius')):
                    # in other words, default is unlimited
                    self.clients_per_round -= 1

    async def send_in_chunks(self, data, sid, client_id) -> None:
        """ Sending a bytes object in fixed-sized chunks to the client. """
        step = 1024 * 256
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        pause_period = 20
        for idx, chunk in enumerate(chunks):
            await self.sio.emit('chunk', {'data': chunk}, room=sid)
            if idx % pause_period == pause_period - 1:
                await asyncio.sleep(0.1)

        await self.sio.emit('payload', {'id': client_id}, room=sid)

    async def send(self, sid, payload, client_id) -> None:
        """ Sending a new data payload to the client using either S3 or socket.io. """
        if self.s3_client is not None:
            payload_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(payload_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
        else:
            payload_key = None
            data_size = 0

            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, sid, client_id)
                    data_size += sys.getsizeof(_data)

            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, sid, client_id)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('payload_done', {
            'id': client_id,
            'obkey': payload_key
        },
                            room=sid)

        logging.info("[Server #%d] Sent %s MB of payload data to client #%d.",
                     os.getpid(), round(data_size / 1024**2, 2), client_id)

    async def client_report_arrived(self, sid, report):
        """ Upon receiving a report from a client. """
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None
        self.client_chunks[sid] = []

    async def client_chunk_arrived(self, sid, data) -> None:
        """ Upon receiving a chunk of data from a client. """
        self.client_chunks[sid].append(data)

    async def client_payload_arrived(self, sid, client_id):
        """ Upon receiving a portion of the payload from a client. """
        assert len(
            self.client_chunks[sid]) > 0 and client_id in self.selected_clients

        payload = b''.join(self.client_chunks[sid])
        _data = pickle.loads(payload)
        self.client_chunks[sid] = []

        if self.client_payload[sid] is None:
            self.client_payload[sid] = _data
        elif isinstance(self.client_payload[sid], list):
            self.client_payload[sid].append(_data)
        else:
            self.client_payload[sid] = [self.client_payload[sid]]
            self.client_payload[sid].append(_data)

    def process_aggregated_weights(self):
        """Process the aggregated weights by testing the accuracy."""

        # Testing the global model accuracy
        model_name = Config().trainer.model_name
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            if 'albert' in model_name:
                logging.info(
                    '[Server #{:d}] Average client perplexity: {:.2f}.'.format(
                        os.getpid(), self.accuracy))
            else:
                logging.info(
                    '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                        os.getpid(), 100 * self.accuracy))
        else:
            # Testing the updated model directly at the server
            self.accuracy = self.trainer.server_test(self.testset,
                                                     mode="async")

            if 'albert' in model_name:
                logging.info(
                    '[Server #{:d}] Global model perplexity: {:.2f}\n'.format(
                        os.getpid(), self.accuracy))
            else:
                logging.info(
                    '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                        os.getpid(), 100 * self.accuracy))

        if hasattr(Config().trainer, 'use_wandb'):
            wandb.log({"accuracy": self.accuracy})

    async def client_payload_done(self, sid, client_id, object_key):
        """ Upon receiving all the payload from a client, eithe via S3 or socket.io. """
        if object_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(
                object_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        current_time = time.perf_counter()
        start_time = self.client_manager.get_start_timestamp(client_id)
        actual_latency = current_time - start_time

        if hasattr(Config().server, 'response_latency_distribution'):
            actual_latency = await self.client_manager.simulate_resp_lat(
                client_id, payload_size, actual_latency)

        if hasattr(Config().server, 'client_selection') \
                and Config().server.client_selection.name == 'oort':
            self.client_manager.update_client_selector(
                name="oort",
                client_id=client_id,
                time_stamp=(self.current_update_version + 1) if hasattr(
                    Config().server, 'asynchronous') else self.current_round,
                utility=self.reports[sid].moving_loss_norm,
                duration=actual_latency)
        elif hasattr(Config().server, 'asynchronous') \
                and (hasattr(Config().server.asynchronous, 'sirius')
                     or hasattr(Config().server.asynchronous, 'fedbuff')):
            self.client_manager.record_training_end(
                client_id=client_id,
                end_time=time.perf_counter()
            )

        if hasattr(Config().server, 'asynchronous') \
             and hasattr(Config().server.asynchronous, 'sirius'):
            self.client_manager.update_utilities_using_loss(
                client_id=client_id,
                loss_norm=self.reports[sid].moving_loss_norm
            )

        if not hasattr(Config().server, 'asynchronous') \
                and hasattr(Config().server, 'overselection') \
                and client_id in self.clients_to_discard:
            self.clients_to_discard.remove(client_id)
            self.selected_clients.remove(client_id)
            logging.info(
                "[Server #%d] Discarding client #%d's payload due to overselection. "
                "Remaining clients to discard: %s.",
                os.getpid(), client_id, self.clients_to_discard)
            return

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        # in case that a client pushes updates twice or for more times
        # while the server has not timely incorporate its old update
        if client_id not in self.payload_done_clients:
            self.payload_done_clients.append(client_id)
            self.updates.append((client_id, self.reports[sid], self.client_payload[sid]))
        else:
            logging.info(f"A {client_id}'s update is discarded due to being lapped by herself.")
            self.payload_done_clients.remove(client_id)  # preserve the arrival order
            self.payload_done_clients.append(client_id)

            idx_to_remove = None
            for idx, update in enumerate(self.updates):
                if update[0] == client_id:
                    idx_to_remove = idx
                    break
            del self.updates[idx_to_remove]
            self.updates.append((client_id, self.reports[sid], self.client_payload[sid]))

        if hasattr(Config().server, 'asynchronous'):
            self.selected_clients.remove(client_id)
            self.available_clients[client_id] = {
                'sid': sid,
                'available_from': time.perf_counter()
            }

            if hasattr(Config().server.asynchronous, 'num_running') and \
                    Config().server.asynchronous.num_running == 'limited' \
                    or hasattr(Config().server.asynchronous, 'sirius'):
                # in other words, default is unlimited
                self.clients_per_round += 1

        elif len(self.updates) > 0:
            if hasattr(Config().server, "overselection"):
                if len(self.updates) < min(self.clients_per_round,
                                           len(self.selected_clients)):
                    return
                else:
                    new_clients_to_discard = set(self.selected_clients) - set(self.payload_done_clients)
                    self.clients_to_discard = self.clients_to_discard.union(new_clients_to_discard)
                    logging.info(
                        "[Server #%d] Overselection: new clients to discard: %s, all clients to discard: %s.",
                        os.getpid(), new_clients_to_discard, self.clients_to_discard)
            elif len(self.updates) < len(self.selected_clients):
                return

            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))

            await self.aggregate_weights(self.updates)
            await self.wrap_up_processing_reports()
            await self.wrap_up()
            await self.select_clients()

    async def client_disconnected(self, sid):
        """ When a client disconnected it should be removed from its internal states. """
        for client_id, client in dict(self.clients).items():
            if client['sid'] == sid:

                if client_id in self.do_want_to_disconnect:
                    del self.clients[client_id]
                    if hasattr(Config().server, 'asynchronous'):
                        if client_id in self.available_clients:
                            del self.available_clients[client_id]

                    self.do_want_to_disconnect.remove(client_id)

                    logging.info(
                        "[Server #%d] Client #%d disconnected and removed from this server.",
                        os.getpid(), client_id)

                    if client_id in self.selected_clients:
                        self.selected_clients.remove(client_id)

                        if not hasattr(Config().server, 'asynchronous') \
                                and len(self.updates) > 0:

                            if hasattr(Config().server, "overselection"):
                                if client_id in self.clients_to_discard:
                                    self.clients_to_discard.remove(client_id)
                                if len(self.updates) < min(self.clients_per_round,
                                                           len(self.selected_clients)):
                                    return
                            elif len(self.updates) < len(self.selected_clients):
                                return

                            logging.info(
                                "[Server #%d] All %d client reports received. Processing.",
                                os.getpid(), len(self.updates))
                            await self.aggregate_weights(self.updates)
                            await self.wrap_up_processing_reports()
                            await self.wrap_up()
                            await self.select_clients()
                else:
                    logging.info(
                        "[Server #%d] Client #%d disconnected but it probably does not mean to. "
                        "Waiting for it to reconnect.", os.getpid(), client_id)

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        # Break the loop when the target accuracy is achieved

        if hasattr(Config().trainer, 'target_accuracy'):
            target_accuracy = Config().trainer.target_accuracy
        else:
            target_accuracy = None

        model_name = Config().trainer.model_name
        if 'albert' in model_name:
            if target_accuracy and 2 < self.accuracy <= target_accuracy:
                logging.info("[Server #%d] Target perplexity reached.", os.getpid())
                await self.close()
        else:
            if target_accuracy and self.accuracy >= target_accuracy:
                logging.info("[Server #%d] Target accuracy reached.", os.getpid())
                await self.close()

        if self.seconds:
            current_time = time.perf_counter()
            if current_time - self.start_time >= self.seconds:
                logging.info("Target end time reached.")
                await self.close()
        elif self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self.close()

    # pylint: disable=protected-access
    async def close(self):
        """Closing the server."""
        logging.info("[Server #%d] Training concluded.", os.getpid())
        self.trainer.save_model()
        await self.close_connections()
        self.parent_pipe.send('exit')
        self.parent_pipe.close()
        sys.exit(0)

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        return server_response

    @abstractmethod
    async def register_client(self, sid, data):
        """Adding a newly arrived client to the list of clients."""
    @abstractmethod
    def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""
    @abstractmethod
    def configure(self):
        """ Configuring the server with initialization work. """
    @abstractmethod
    async def process_reports(self) -> None:
        """ Process a client report. """
    @abstractmethod
    def choose_clients(self) -> list:
        """ Choose a subset of the clients to participate in each round. """
