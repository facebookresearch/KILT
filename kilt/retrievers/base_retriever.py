# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from abc import ABC, abstractmethod
from kilt.configs import retriever


class Retriever(ABC):
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_default_config(cls, name):
        import importlib.resources

        config = json.loads(
            importlib.resources.read_text(
                retriever, "default_{name}.json".format(name=name)
            )
        )
        return cls(name, **config)

    @classmethod
    def from_config_file(cls, name, config_file):
        with open(config_file, "r") as cf:
            config = json.load(cf)
        return cls(name, **config)

    @classmethod
    def from_config_string(cls, name, config_string):
        config = json.loads(config_string)
        return cls(name, **config)

    @abstractmethod
    def fed_data(self, queries_data, topk, logger=None):
        """
        fed all data to the retriever, that will take care of batchify it
        each element in queries_data has an id and a query
        """
        pass

    @abstractmethod
    def run(self):
        """
        get the retrieved documents for all the fed data
        """
        pass
