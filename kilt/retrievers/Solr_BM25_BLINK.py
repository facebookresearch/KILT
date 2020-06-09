import multiprocessing
from multiprocessing.pool import ThreadPool

import pysolr
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import re

import kilt.utils as utils
from kilt.retrievers.base_retriever import Retriever


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
    ner_model = arguments["ner_model"]
    query_arguments = arguments["query_arguments"]

    if id == 0:
        iter_ = tqdm(queries_data)
    else:
        iter_ = queries_data

    result_doc_ids = []
    result_doc_scores = []
    result_query_id = []

    for query_element in iter_:

        query = solr_escape(str(query_element["query"]))
        result_query_id.append(query_element["id"])

        # get all entities in query
        sent = Sentence(query, use_tokenizer=True)
        ner_model.predict(sent)
        sent_mentions = sent.to_dict(tag_type="ner")["entities"]

        doc_ids = []
        doc_scores = []

        for mention in sent_mentions:
            entity = solr_escape(str(mention))

            solr_query = "title:( {} ) OR aliases:( {} ) OR sent_desc_1:( {} )^0.5".format(
                entity, entity, query
            )
            results = solr.search(solr_query, **query_arguments)

            doc_ids.extend([cand["id"] for cand in results.docs])
            doc_scores.extend([cand["score"] for cand in results.docs])

        if len(doc_ids) == 0:
            # consider only descritpion when no entities are found
            # TODO: try sent_desc_1 -> desc
            solr_query = "sent_desc_1:( {} )".format(query)
            results = solr.search(solr_query, **query_arguments)

            doc_ids.extend([cand["id"] for cand in results.docs])
            doc_scores.extend([cand["score"] for cand in results.docs])

        if len(sent_mentions) > 1:
            # merge predictions when multiple entities are found
            sorted_ids = []
            sorted_scores = []
            for y, x in sorted(zip(doc_scores, doc_ids), reverse=True):
                if x not in sorted_ids:
                    sorted_ids.append(x)
                    sorted_scores.append(y)
            doc_ids = sorted_ids[:topk]
            doc_scores = sorted_scores[:topk]

        result_doc_ids.append(doc_ids)
        result_doc_scores.append(doc_scores)

    return result_doc_ids, result_doc_scores, result_query_id


class Solr(Retriever):
    def __init__(self, name, solr_address, fl, rows, defType, bf, num_threads):

        super().__init__(name)
        self.solr_address = solr_address
        self.query_arguments = {
            "fl": fl,
            "rows": rows,
            "defType": defType,
            "bf": bf,
        }

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
                    "ner_model": SequenceTagger.load("ner"),
                    "query_arguments": self.query_arguments,
                }
            )

    def fed_data(self, queries_data, topk, logger=None):

        self.query_arguments["rows"] = topk

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

        return (
            all_doc_id,
            all_doc_scores,
            all_query_id,
        )
