"""
The registry that contains all available federated learning client managers.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging
from collections import OrderedDict
from plato.config import Config

from plato.client_managers import (base, sirius, fedbuff)

registered_clients = OrderedDict([('base', base.ClientManager),
                                  ('sirius', sirius.ClientManager),
                                  ('fedbuff', fedbuff.ClientManager)])


def get():
    """Get an instance of the client manager"""
    manager_type = "base"
    if hasattr(Config().server, 'asynchronous'):
        if hasattr(Config().server.asynchronous, 'sirius'):
            manager_type = "sirius"
        elif hasattr(Config().server.asynchronous, 'fedbuff'):
            manager_type = "fedbuff"

    if manager_type in registered_clients:
        logging.info("Client manager: %s", manager_type)
        registered_manager = registered_clients[manager_type]()
    else:
        raise ValueError('No such client manager: {}'.format(manager_type))

    return registered_manager
