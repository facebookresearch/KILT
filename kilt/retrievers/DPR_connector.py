# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import argparse
import glob
import pickle

from dpr.utils.model_utils import (
    load_states_from_checkpoint,
    setup_for_distributed_mode,
    get_model_obj,
)
from dpr.options import set_encoder_params_from_state
from dpr.models import init_biencoder_components
from dense_retriever import (
    DenseRetriever,
    parse_qa_csv_file,
    load_passages,
    iterate_encoded_files,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

from kilt.configs import retriever
import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever


class DPR(Retriever):
    def __init__(self, name, **config):
        super().__init__(name)

        self.args = argparse.Namespace(**config)
        saved_state = load_states_from_checkpoint(self.args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, self.args)
        tensorizer, encoder, _ = init_biencoder_components(
            self.args.encoder_model_type, self.args, inference_only=True
        )
        encoder = encoder.question_model
        encoder, _ = setup_for_distributed_mode(
            encoder,
            None,
            self.args.device,
            self.args.n_gpu,
            self.args.local_rank,
            self.args.fp16,
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)

        prefix_len = len("question_model.")
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("question_model.")
        }
        model_to_load.load_state_dict(question_encoder_state, strict=False)
        vector_size = model_to_load.get_out_size()

        # index all passages
        ctx_files_pattern = self.args.encoded_ctx_file
        input_paths = glob.glob(ctx_files_pattern)

        index_buffer_sz = self.args.index_buffer
        if self.args.hnsw_index:
            index = DenseHNSWFlatIndexer(vector_size)
            index.deserialize_from(self.args.hnsw_index_path)
        else:
            index = DenseFlatIndexer(vector_size)
            index.index_data(input_paths)

        self.retriever = DenseRetriever(
            encoder, self.args.batch_size, tensorizer, index
        )

        # not needed for now
        self.all_passages = load_passages(self.args.ctx_file)

        self.KILT_mapping = None
        if self.args.KILT_mapping:
            self.KILT_mapping = pickle.load(open(self.args.KILT_mapping, "rb"))

    def feed_data(
        self,
        queries_data,
        ent_start_token=utils.ENT_START,
        ent_end_token=utils.ENT_START,
        logger=None,
    ):

        # get questions & answers
        self.questions = [
            x["query"].replace(ent_start_token, "").replace(ent_end_token, "").strip()
            for x in queries_data
        ]
        self.query_ids = [x["id"] for x in queries_data]

    def run(self):

        questions_tensor = self.retriever.generate_question_vectors(self.questions)
        top_ids_and_scores = self.retriever.get_top_docs(
            questions_tensor.numpy(), self.args.n_docs
        )

        provenance = {}

        for record, query_id in zip(top_ids_and_scores, self.query_ids):
            top_ids, scores = record
            element = []

            # sort by score in descending order
            for score, id in sorted(zip(scores, top_ids), reverse=True):

                text = self.all_passages[id][0]
                index = self.all_passages[id][1]

                wikipedia_id = None
                if self.KILT_mapping:
                    # passages indexed by wikipedia title - mapping needed
                    title = index
                    if title in self.KILT_mapping:
                        wikipedia_id = self.KILT_mapping[title]
                else:
                    # passages indexed by wikipedia id
                    wikipedia_id = index

                element.append(
                    {
                        "score": str(score),
                        "text": str(text),
                        "wikipedia_title": str(index),
                        "wikipedia_id": str(wikipedia_id),
                    }
                )

            assert query_id not in provenance
            provenance[query_id] = element

        return provenance
