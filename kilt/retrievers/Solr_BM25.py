import multiprocessing
from multiprocessing.pool import ThreadPool

import pickle
from tqdm import tqdm
import re

import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever
from kilt.knowledge_source import KnowledgeSource


ESCAPE_CHARS_RE = re.compile(r'(?<!\\)(?P<char>[&|+\-!(){}[\]\/^"~*?:])')


def solr_escape(string):
    if (string == "OR") or (string == "AND"):
        return string.lower()

    interior = r"\s+(OR|AND)\s+"
    start = r"^(OR|AND) "
    end = r" (OR|AND)$"

    string = re.sub(interior, lambda x: x.group(0).lower(), string)
    string = re.sub(start, lambda x: x.group(0).lower(), string)
    string = re.sub(end, lambda x: x.group(0).lower(), string)

    return ESCAPE_CHARS_RE.sub(r"\\\g<char>", string)


def _get_predictions_thread(arguments):

    id = arguments["id"]
    queries_data = arguments["queries_data"]
    topk = arguments["topk"]
    solr = arguments["solr"]
    logger = arguments["logger"]
    query_arguments = arguments["query_arguments"]
    KILT_mapping = arguments["KILT_mapping"]

    if id == 0:
        iter_ = tqdm(queries_data)
    else:
        iter_ = queries_data

    result_doc_ids = []
    result_doc_scores = []
    result_query_id = []

    show_example = True

    for query_element in iter_:

        query = solr_escape(str(query_element["query"]))
        result_query_id.append(query_element["id"])

        solr_query = "text:( {} )".format(query)

        # print(solr_query)

        results = solr.search(solr_query, **query_arguments)

        #  print(len(results))

        doc_ids = []
        doc_scores = []

        for cand in results.docs:

            # passages indexed by wikipedia title - mapping needed
            title = cand["wikipedia_title"].strip()

            if title[0] == '"':
                title = title[1:]
            if title[-1] == '"':
                title = title[:-1]

            if title in KILT_mapping:
                wikipedia_id = KILT_mapping[title]

                if wikipedia_id not in doc_ids:
                    doc_ids.append(wikipedia_id)
                    doc_scores.append(cand["score"])

        #  print(len(doc_ids))
        #  input("...")

        result_doc_ids.append(doc_ids)
        result_doc_scores.append(doc_scores)

    return result_doc_ids, result_doc_scores, result_query_id


class Solr(Retriever):
    def __init__(
        self, name, solr_address, fl, rows, defType, num_threads, KILT_mapping
    ):
        import pysolr

        super().__init__(name)
        self.solr_address = solr_address
        self.query_arguments = {
            "fl": fl,
            "rows": rows,
            "defType": defType,
        }

        self.KILT_mapping = pickle.load(open(KILT_mapping, "rb"))

        self.num_threads = min(num_threads, int(multiprocessing.cpu_count()))

        # initialize a solr connection and a flair model per thread
        self.arguments = []

        for id in tqdm(range(self.num_threads)):

            self.arguments.append(
                {
                    "id": id,
                    "solr": pysolr.Solr(
                        self.solr_address, always_commit=True, timeout=100
                    ),
                    "KILT_mapping": self.KILT_mapping,
                    "query_arguments": self.query_arguments,
                }
            )

    def fed_data(self, queries_data, topk, logger=None):

        chunked_queries = utils.chunk_it(queries_data, self.num_threads)

        for idx, arg in enumerate(self.arguments):
            arg["queries_data"] = chunked_queries[idx]
            arg["topk"] = topk
            arg["logger"] = logger

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_get_predictions_thread, self.arguments)

        all_doc_id = []
        all_doc_scores = []
        all_query_id = []

        for x in results:
            i, s, q = x
            all_doc_id.extend(i)
            all_doc_scores.extend(s)
            all_query_id.extend(q)

        pool.terminate()
        pool.join()

        meta = None

        return all_doc_id, all_doc_scores, all_query_id, meta
