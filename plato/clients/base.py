"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import asyncio
import logging
import os
import pickle
import sys
import uuid
import time
from abc import abstractmethod
from dataclasses import dataclass
import socketio
import multiprocessing as mp
from plato.config import Config
from plato.utils import s3


@dataclass
class Report:
    """Client report, to be sent to the federated learning server."""
    num_samples: int
    accuracy: float


class ClientEvents(socketio.AsyncClientNamespace):
    """ A custom namespace for socketio.AsyncServer. """
    def __init__(self, namespace, plato_client):
        super().__init__(namespace)
        self.plato_client = plato_client
        self.client_id = plato_client.client_id

    #pylint: disable=unused-argument
    async def on_connect(self):
        """ Upon a new connection to the server. """
        logging.info("[Client #%d] Connected to the server.", self.client_id)
        if self.plato_client.disconnect_halfway:
            await self.plato_client.sio.emit('client_alive',
                                             {'id': self.client_id})
            self.plato_client.disconnect_halfway = False

    # pylint: disable=protected-access
    async def on_disconnect(self):
        """ Upon a disconnection event. """
        logging.info(
            "[Client #%d] The connection is dropped. "
            "Try to reconnect...", self.client_id)
        self.plato_client.disconnect_halfway = True

    # pylint: disable=protected-access
    async def on_shutdown(self):
        """ Upon a disconnection event. """
        logging.info("[Client #%d] The server disconnected the connection.",
                     self.client_id)
        # instead of using os._exit(0), sys.exit can
        # clean up leaked semaphore objects at shutdown
        sys.exit(0)

    async def on_connect_error(self, data):
        """ Upon a failed connection attempt to the server. """
        logging.info("[Client #%d] A connection attempt to the server failed.",
                     self.client_id)

    async def on_payload_to_arrive(self, data):
        """ New payload is about to arrive from the server. """
        await self.plato_client.payload_to_arrive(data['response'])

    async def on_chunk(self, data):
        """ A chunk of data from the server arrived. """
        await self.plato_client.chunk_arrived(data['data'])

    async def on_payload(self, data):
        """ A portion of the new payload from the server arrived. """
        await self.plato_client.payload_arrived(data['id'])

    async def on_payload_done(self, data):
        """ All of the new payload sent from the server arrived. """
        await self.plato_client.payload_done(data['id'], data['obkey'])

    async def on_resend_request(self, data):
        logging.info("[Client #%d] The server told me to "
                     "resend if necessary.",
                     self.client_id)
        await self.plato_client.resend_payload()

    async def on_update_concentration(self, data):
        logging.info(f"[Client #%d] Update concentration.",
                     self.client_id)
        await self.plato_client.update_concentration(
            data['concentration'])

    async def on_do_data_corruption(self, data):
        logging.info(f"[Client #%d] To simulate data corruption.",
                     self.client_id)
        await self.plato_client.do_data_corruption()

    async def on_update_partition_size(self, data):
        logging.info(f"[Client #%d] Update partition_size.",
                     self.client_id)
        await self.plato_client.update_partition_size(
            data['partition_size'])


class Client:
    """ A basic federated learning client. """
    def __init__(self) -> None:
        self.client_id = Config().args.id
        self.sio = None
        self.chunks = []
        self.server_payload = None
        self.data_loaded = False  # is training data already loaded from the disk?
        self.s3_client = None
        self.disconnect_halfway = False
        self.model_trained = False
        self.report_backup = None
        self.payload_backup = None

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            self.edge_server_id = None

            assert hasattr(Config().algorithm, 'total_silos')

    async def training_coro(self):
        check_interval = 1
        while True:
            if self.trainer.async_training_ended():
                report, payload = await self.train()
                await self.payload_done_part_two(self.client_id, report,
                                                 payload)
                self.trainer.reset_async_training_status()
            await asyncio.sleep(check_interval)

    async def start_client(self) -> None:
        """ Startup function for a client. """

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                self.edge_server_id = int(
                    Config().clients.per_round) + (self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            else:
                self.edge_server_id = int(Config().clients.total_clients) + (
                    self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            logging.info("[Client #%d] Contacting Edge server #%d.",
                         self.client_id, self.edge_server_id)
        else:
            await asyncio.sleep(5)
            logging.info("[Client #%d] Contacting the central server.",
                         self.client_id)

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(
            ClientEvents(namespace='/', plato_client=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        if hasattr(Config().server, 'use_https'):
            uri = 'https://{}'.format(Config().server.address)
        else:
            uri = 'http://{}'.format(Config().server.address)

        if hasattr(Config().server, 'port'):
            # If we are not using a production server deployed in the cloud
            if hasattr(Config().algorithm,
                       'cross_silo') and not Config().is_edge_server():
                uri = '{}:{}'.format(
                    uri,
                    int(Config().server.port) + int(self.edge_server_id))
            else:
                uri = '{}:{}'.format(uri, Config().server.port)

        logging.info("[Client #%d] Connecting to the server at %s.",
                     self.client_id, uri)

        connection_gap = 1
        while True:
            try:
                await self.sio.connect(uri)
            except socketio.exceptions.ConnectionError as e:
                logging.info(
                    "The server is not online (%s). "
                    "Try to reconnect %ss later.", e, connection_gap)
                time.sleep(connection_gap)
            except Exception as e:
                raise e
            else:
                break

        self.load_data()  # need to know trainset_size
        await self.sio.emit('client_alive', {
            'id': self.client_id,
            'trainset_size': self.sampler.trainset_size()
        })

        if hasattr(Config().clients, 'async_training') \
                and Config().clients.async_training is True:
            current_loop = asyncio.get_event_loop()
            current_loop.create_task(self.training_coro())

        logging.info("[Client #%d] Waiting to be selected.", self.client_id)
        await self.sio.wait()

    async def payload_to_arrive(self, response) -> None:
        """ Upon receiving a response from the server. """
        self.process_server_response(response)

        # Update (virtual) client id for client, trainer and algorithm
        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            self.client_id = response['id']
            self.configure()

        logging.info("[Client #%d] Selected by the server.", self.client_id)

        if not self.data_loaded:
            self.load_data()

        self.model_trained = False
        self.report_backup = None
        self.payload_backup = None

    async def chunk_arrived(self, data) -> None:
        """ Upon receiving a chunk of data from the server. """
        self.chunks.append(data)

    async def payload_arrived(self, client_id) -> None:
        """ Upon receiving a portion of the new payload from the server. """
        assert client_id == self.client_id

        payload = b''.join(self.chunks)
        _data = pickle.loads(payload)
        self.chunks = []

        if self.server_payload is None:
            self.server_payload = _data
        elif isinstance(self.server_payload, list):
            self.server_payload.append(_data)
        else:
            self.server_payload = [self.server_payload]
            self.server_payload.append(_data)

    async def payload_done(self, client_id, object_key) -> None:
        """ Upon receiving all the new payload from the server. """
        payload_size = 0

        if object_key is None:
            if isinstance(self.server_payload, list):
                for _data in self.server_payload:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            elif isinstance(self.server_payload, dict):
                for key, value in self.server_payload.items():
                    payload_size += sys.getsizeof(pickle.dumps({key: value}))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.server_payload))
        else:
            self.server_payload = self.s3_client.receive_from_s3(object_key)
            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

        assert client_id == self.client_id

        logging.info(
            "[Client #%d] Received %s MB of payload data from the server.",
            client_id, round(payload_size / 1024**2, 2))

        self.load_payload(self.server_payload)
        self.server_payload = None

        report, payload = await self.train()
        if not (hasattr(Config().clients, 'async_training') \
                and Config().clients.async_training is True):
            await self.payload_done_part_two(client_id, report, payload)

    async def payload_done_part_two(self, client_id, report, payload) -> None:
        if Config().is_edge_server():
            logging.info(
                "[Server #%d] Model aggregated on edge server (client #%d).",
                os.getpid(), client_id)
        else:
            logging.info("[Client #%d] Model trained.", client_id)
        self.model_trained = True
        self.report_backup = report
        self.payload_backup = payload

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {'report': pickle.dumps(report)})

        # Sending the client training payload to the server
        await self.send(payload)

    async def resend_payload(self):
        if self.model_trained:
            logging.info("[Client #%d] Resending payload.", self.client_id)
            await self.sio.emit('client_report',
                                {'report': pickle.dumps(self.report_backup)})
            await self.send(self.payload_backup)

    async def send_in_chunks(self, data) -> None:
        """ Sending a bytes object in fixed-sized chunks to the client. """
        step = 1024 * 256
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        pause_period = 20
        for idx, chunk in enumerate(chunks):
            await self.sio.emit('chunk', {'data': chunk, 'idx': idx})
            if idx % pause_period == pause_period - 1:
                await asyncio.sleep(0.1)

        await self.sio.emit('client_payload', {'id': self.client_id})

    async def send(self, payload) -> None:
        """Sending the client payload to the server using either S3 or socket.io."""
        if self.s3_client != None:
            unique_key = uuid.uuid4().hex[:6].upper()
            payload_key = f'client_payload_{self.client_id}_{unique_key}'
            self.s3_client.send_to_s3(payload_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
        else:
            payload_key = None
            if isinstance(payload, list):
                data_size: int = 0

                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data)
                    data_size += sys.getsizeof(_data)
            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('client_payload_done', {
            'id': self.client_id,
            'obkey': payload_key
        })

        logging.info("[Client #%d] Sent %s MB of payload data to the server.",
                     self.client_id, round(data_size / 1024**2, 2))

    async def do_data_corruption(self):
        self.trainset = self.datasource\
            .simulate_data_corruption(self.client_id)

    async def update_concentration(self, concentration):
        # only make sense with dirichlet sampler
        self.sampler.update_concentration(concentration)

    async def update_partition_size(self, partition_size):
        # only make sense with dirichlet sampler
        self.sampler.update_partition_size(partition_size)

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
    @abstractmethod
    def configure(self) -> None:
        """Prepare this client for training."""
    @abstractmethod
    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
    @abstractmethod
    def load_payload(self, server_payload) -> None:
        """Loading the payload onto this client."""
    @abstractmethod
    async def train(self):
        """The machine learning training workload on a client."""
