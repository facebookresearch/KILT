# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing
from multiprocessing.pool import ThreadPool
import json

from tqdm import tqdm
import jnius_config

import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever


def _run_thread(arguments):
    idz = arguments["id"]
    index = arguments["index"]
    k = arguments["k"]
    data = arguments["data"]

    # BM25 parameters #TODO
    # bm25_a = arguments["bm25_a"]
    # bm25_b = arguments["bm25_b"]
    # searcher.set_bm25(bm25_a, bm25_b)

    from pyserini.search import SimpleSearcher

    searcher = SimpleSearcher(index)

    _iter = data
    if idz == 0:
        _iter = tqdm(data)

    provenance = {}
    for x in _iter:
        query_id = x["id"]
        query = (
            x["query"].replace(utils.ENT_END, "").replace(utils.ENT_START, "").strip()
        )

        hits = searcher.search(query, k)

        element = []
        for y in hits:
            try:
                doc_data = json.loads(str(y.docid).strip())
                doc_data["score"] = y.score
                doc_data["text"] = str(y.raw).strip()
                element.append(doc_data)
            except Exception as e:
                print(e)
                element.append(
                    {
                        "score": y.score,
                        "text": str(y.raw).strip(),
                        "title": y.docid,
                    }
                )
        provenance[query_id] = element

    return provenance


class BM25(Retriever):
    def __init__(self, name, index, k, num_threads, Xms=None, Xmx=None):
        super().__init__(name)

        if Xms and Xmx:
            # to solve Insufficient memory for the Java Runtime Environment
            jnius_config.add_options(
                "-Xms{}".format(Xms), "-Xmx{}".format(Xmx), "-XX:-UseGCOverheadLimit"
            )
            print("Configured options:", jnius_config.get_options())

        self.num_threads = min(num_threads, int(multiprocessing.cpu_count()))

        # initialize a ranker per thread
        self.arguments = []
        for id in tqdm(range(self.num_threads)):
            self.arguments.append(
                {
                    "id": id,
                    "index": index,
                    "k": k,
                }
            )

    def feed_data(self, queries_data, logger=None):

        chunked_queries = utils.chunk_it(queries_data, self.num_threads)

        for idx, arg in enumerate(self.arguments):
            arg["data"] = chunked_queries[idx]

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_run_thread, self.arguments)

        provenance = {}
        for x in results:
            provenance.update(x)
        pool.terminate()
        pool.join()

        return provenance
