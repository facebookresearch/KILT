# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
from pyserini.search import SimpleSearcher

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
    def __init__(self, name, index, k, num_threads):
        super().__init__(name)

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

    def fed_data(self, queries_data, logger=None):

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
