# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import pickle
import zlib
from omegaconf import OmegaConf
from tqdm import tqdm

from dpr.models import init_biencoder_components
from dpr.options import setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dense_retriever import DenseRPCRetriever

import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DPR(Retriever):
    def __init__(self, name, cfg):
        super().__init__(name)

        cfg = setup_cfg_gpu(cfg)

        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, encoder, _ = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, inference_only=True
        )

        encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(
            encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")

        encoder_prefix = "question_model."
        prefix_len = len(encoder_prefix)

        logger.info("Encoder state prefix %s", encoder_prefix)
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(encoder_prefix)
            and key != "question_model.embeddings.position_ids"
        }
        model_to_load.load_state_dict(question_encoder_state, strict=False)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        self.retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
        self.retriever.load_index(cfg.rpc_index_id)

        self.KILT_mapping = None
        if cfg.KILT_mapping:
            self.KILT_mapping = dict(pickle.load(open(cfg.KILT_mapping, "rb")))

        self.rpc_meta_compressed = cfg.rpc_meta_compressed
        self.cfg = cfg

    @classmethod
    def from_config_file(cls, name, config_file):
        cfg = OmegaConf.load(config_file)
        return cls(name, cfg)

    @classmethod
    def process_query(cls, x, ent_start_token, ent_end_token):
        return x["query"].replace(ent_start_token, "").replace(
            ent_end_token, ""
        ).strip() + ("?" if not x["query"].endswith("?") else "")

    def feed_data(
        self,
        queries_data,
        ent_start_token=utils.ENT_START,
        ent_end_token=utils.ENT_START,
        logger=None,
    ):

        # get questions & answers
        self.questions = [
            DPR.process_query(x, ent_start_token, ent_end_token) for x in queries_data
        ]
        self.query_ids = [x["id"] for x in queries_data]

    def run(self):

        dup_multiplier = 1
        questions_tensor = self.retriever.generate_question_vectors(self.questions)
        top_ids_and_scores = self.retriever.get_top_docs(
            questions_tensor.numpy(), dup_multiplier * self.cfg.n_docs, search_batch=256
        )

        provenance = {}

        for record, query_id in tqdm(zip(top_ids_and_scores, self.query_ids)):
            element = []
            docs_meta, scores = record

            cnt = 0
            for score, meta in zip(scores, docs_meta):
                if cnt >= self.cfg.n_docs:
                    break
                doc_id, text, title = meta[:3]
                wikipedia_id = (
                    self.KILT_mapping[int(doc_id)]
                    if self.KILT_mapping and (int(doc_id) in self.KILT_mapping)
                    else None
                )

                element.append(
                    {
                        "score": str(score),
                        "text": str(zlib.decompress(text).decode())
                        if self.rpc_meta_compressed
                        else text,
                        "wikipedia_title": str(zlib.decompress(title).decode())
                        if self.rpc_meta_compressed
                        else title,
                        "wikipedia_id": str(wikipedia_id),
                        "doc_id": str(doc_id),
                    }
                )
                cnt += 1

            assert query_id not in provenance
            provenance[query_id] = element

        return provenance
