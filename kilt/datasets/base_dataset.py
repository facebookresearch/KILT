# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import json

from abc import ABC, abstractmethod

from kilt.configs import mapping


class Dataset(ABC):
    def __init__(self, name):
        self.name = name
        self.output_file = None
        self.max_chunks = None

    @classmethod
    def from_default_config(cls, name):
        config = json.loads(
            importlib.resources.read_text(
                mapping, "default_{name}.json".format(name=name)
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

    def get_chunks(self, num_chunks):
        """
        Retruns a list of chunks of the dataset.
        """
        pass

    @abstractmethod
    def process_chunk(self, chunk, ks, chunk_id):
        """
        Processes a single chunk of the dataset. Maps each line in the
        chunk into the kilt format. Returns a list of mapped entries and
        optionally metadata.
        """
        pass

    @abstractmethod
    def postprocess_metadata(self, metadata):
        pass
