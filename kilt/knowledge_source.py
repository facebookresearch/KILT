# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pymongo import MongoClient

DEFAULT_MONGO_CONNECTION_STRING = "mongodb://127.0.0.1:27017/admin"


class KnowledgeSource:
    def __init__(
        self,
        mongo_connection_string=None,
        database="kilt",
        collection="knowledgesource",
    ):
        if not mongo_connection_string:
            mongo_connection_string = DEFAULT_MONGO_CONNECTION_STRING
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client[database][collection]

    def get_all_pages_cursor(self):
        cursor = self.db.find({})
        return cursor

    def get_num_pages(self):
        return self.db.count()

    def get_page_by_id(self, wikipedia_id):
        page = self.db.find_one({"_id": str(wikipedia_id)})
        return page

    def get_page_by_title(self, wikipedia_title, attempt=0):
        page = self.db.find_one({"wikipedia_title": str(wikipedia_title)})
        return page
