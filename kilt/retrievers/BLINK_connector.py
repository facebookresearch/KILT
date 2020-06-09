import argparse

from flair.models import SequenceTagger
from flair.data import Sentence
import blink.main_dense as main_dense
import logging
import pickle

from kilt.retrievers.base_retriever import Retriever
from kilt.knowledge_source import KnowledgeSource

WIKIPEDIA_TITLE2ID = "/checkpoint/fabiopetroni/KILT/Wikipedia_title2id.p"


class BLINK(Retriever):
    def __init__(self, name, **config):
        super().__init__(name)
        self.args = argparse.Namespace(**config)

        self.logger = logging.getLogger("KILT")

        self.models = main_dense.load_models(self.args, logger=self.logger)

        # self.ks = KnowledgeSource()

        self.ner_model = SequenceTagger.load("ner")

        self.cache_pages = {}

        self.Wikipedia_title2id = pickle.load(open(WIKIPEDIA_TITLE2ID, "rb"))

    def fed_data(
        self,
        queries_data,
        topk,
        ent_start_token="[START_ENT]",
        ent_end_token="[END_ENT]",
        logger=None,
    ):
        if logger:
            self.logger = logger

        wikipedia_id2local_id = self.models[8]

        self.test_data = []
        for element in queries_data:

            query = element["query"]

            if ent_start_token in query and ent_end_token in query:
                split1 = query.split(ent_start_token)
                assert len(split1) == 2
                left = split1[0]
                split2 = split1[1].split(ent_end_token)
                assert len(split2) == 2
                mention = split2[0]
                right = split2[1]

                record = {
                    "id": element["id"],
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": left.strip().lower(),
                    "mention": mention.strip().lower(),
                    "context_right": right.strip().lower(),
                }
                self.test_data.append(record)
            else:

                # Apply a NER system
                sent = Sentence(query, use_tokenizer=True)
                self.ner_model.predict(sent)
                sent_mentions = sent.to_dict(tag_type="ner")["entities"]

                if len(sent_mentions) == 0:
                    # no mention
                    record = {
                        "id": element["id"],
                        "label": "unknown",
                        "label_id": -1,
                        "context_left": query.strip().lower(),
                        "mention": "",
                        "context_right": "",
                    }
                    self.test_data.append(record)

                else:
                    # create a record for each mention detected
                    for hit in sent_mentions:
                        left = query[: int(hit["start_pos"])].strip()
                        mention = hit["text"]
                        right = query[int(hit["end_pos"]) :].strip()

                        record = {
                            "id": element["id"],
                            "label": "unknown",
                            "label_id": -1,
                            "context_left": left.strip().lower(),
                            "mention": mention.strip().lower(),
                            "context_right": right.strip().lower(),
                        }
                        self.test_data.append(record)

    def run(self):
        (
            biencoder_accuracy,
            recall_at,
            crossencoder_normalized_accuracy,
            overall_unormalized_accuracy,
            num_datapoints,
            predictions,
            scores,
        ) = main_dense.run(
            self.args, self.logger, *self.models, test_data=self.test_data
        )

        # aggregate multiple records for the same datapoint
        print("aggregate multiple records for the same datapoint", flush=True)
        id_2_results = {}
        for r, p, s in zip(self.test_data, predictions, scores):

            if r["id"] not in id_2_results:
                id_2_results[r["id"]] = {"predictions": [], "scores": []}
            id_2_results[r["id"]]["predictions"].extend(p)
            id_2_results[r["id"]]["scores"].extend(s)

        all_doc_id = []
        all_query_id = []
        all_scores = []

        meta = {}

        for id, results in id_2_results.items():

            element = {"id": str(id), "retrieved": []}

            # merge predictions when multiple entities are found
            sorted_titles = []
            sorted_scores = []
            for y, x in sorted(
                zip(results["scores"], results["predictions"]), reverse=True
            ):
                if x not in sorted_titles:
                    sorted_titles.append(x)
                    sorted_scores.append(y)

            local_doc_id = []
            for e_title, score in zip(sorted_titles, sorted_scores):

                if e_title not in self.Wikipedia_title2id:
                    print(
                        "WARNING: title: {} not recognized".format(e_title), flush=True
                    )
                else:

                    """
                    if e_title in self.cache_pages:
                        page = self.cache_pages[e_title]
                    else:
                        page = self.ks.get_page_by_title(e_title)
                        self.cache_pages[e_title] = page
                    
                    wikipedia_id = page["wikipedia_id"]
                    """

                    wikipedia_id = self.Wikipedia_title2id[e_title]

                    local_doc_id.append(wikipedia_id)

                    element["retrieved"].append(
                        {
                            "score": str(score),
                            # "text": page["text"],
                            "wikipedia_title": str(e_title),
                            "wikipedia_id": str(wikipedia_id),
                        }
                    )
            all_doc_id.append(local_doc_id)
            all_scores.append(sorted_scores)
            all_query_id.append(id)
            meta[id] = element["retrieved"]

        return all_doc_id, all_scores, all_query_id, meta
