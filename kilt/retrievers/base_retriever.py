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
    def feed_data(self, queries_data, logger=None):
        """
        fed all data to the retriever, that will take care of batchify it
        each element in queries_data has an id and a query

        Args:
        queries_data (list): list of dicts with two fields: (1) 'id' -> id of the query: (2) 'query' -> text of the query

        Example:
        queries_data = [
            {'id': '-4203908294749842710', 'query': 'what is the definition of bcc in email'},
            ...
        ]
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        get the retrieved documents for all the fed data
        return all_doc_id, all_scores, all_query_id, provenance

        Returns
        -------
        provenance: dictionary with retrieval result, the keys should match the query id in input

        Example:
        provenance: {
            '-4203908294749842710': [
                {"score": "179.01215", "text": "permit the use of a program-external editor. The email clients will perform formatting according to RFC 5322 for headers and body, and MIME for non-textual content and attachments. Headers include the destination fields, \"To\", \"Cc\" (short for \"Carbon copy\"), and \"Bcc\" (\"Blind carbon copy\"), and the originator fields \"From\" which is the message's author(s), \"Sender\" in case there are more authors, and \"Reply-To\"", "wikipedia_title": "Email client", "wikipedia_id": "43478"},
                {"score": "184.6643", "text": "this example, the conversation parts are prefixed with \"S:\" and \"C:\", for \"server\" and \"client\", respectively; these labels are not part of the exchange.) After the message sender (SMTP client) establishes a reliable communications channel to the message receiver (SMTP server), the session is opened with a greeting by the server, usually containing its fully qualified domain name (FQDN), in this case \"smtp.example.com\". The client initiates its dialog by responding with a", "wikipedia_title": "Simple Mail Transfer Protocol", "wikipedia_id": "27675"},
                ...
                ],
            ...
        }
        """
        raise NotImplementedError
