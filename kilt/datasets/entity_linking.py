# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import sys
import uuid

import uuid
from tqdm import tqdm

import kilt.kilt_utils as utils
from kilt.knowledge_source import KnowledgeSource
from kilt.datasets.base_dataset import Dataset


def convert_to_KILT_format(
    questions,
    ks,
    id_filter_positive,
    id_filter_negative,
    max_input_lenght=256,
    ent_start_token="[START_ENT]",
    ent_end_token="[END_ENT]",
):
    data = []
    for q in questions:

        if id_filter_positive:
            if id_filter_positive not in q["id"]:
                continue

        if id_filter_negative:
            if id_filter_negative in q["id"]:
                continue

        page = ks.get_page_from_url(q["Wikipedia_URL"])

        if page:
            left_context = q["left_context"].copy()
            right_context = q["right_context"].copy()

            left = " ".join(left_context).strip()
            text_mention = q["mention"].strip()
            right = " ".join(right_context).strip()

            # create input text
            # balance left and right context
            input_text = (
                left
                + " "
                + ent_start_token
                + " "
                + text_mention
                + " "
                + ent_end_token
                + " "
                + right
            )
            tokens = input_text.split()
            while (
                len(tokens) >= max_input_lenght - 2
            ):  # 2 = ent_start_token + ent_end_token
                offset = max(1, int((len(tokens) - max_input_lenght) / 2))
                len_left = len(left.split())
                len_right = len(right.split())
                if len_left > len_right:
                    left_context = left_context[offset:]
                    left = " ".join(left_context).strip()
                else:
                    right_context = right_context[:offset]
                    right = " ".join(right_context).strip()
                # udpate tokens
                input_text = (
                    left
                    + " "
                    + ent_start_token
                    + " "
                    + text_mention
                    + " "
                    + ent_end_token
                    + " "
                    + right
                )
                tokens = input_text.split()

            datapoint = {
                "id": str(uuid.uuid4()) + "_" + str(q["id"]),
                "input": input_text,
                "output": [
                    {
                        "answer": page["wikipedia_title"],
                        "provenance": [
                            # list of relevant WikipediaPages / Spans as provenance for the answer from the ks
                            {
                                "wikipedia_id": page["wikipedia_id"],
                                "title": page["wikipedia_title"],
                            }
                        ],
                    }
                ],
                "meta": {
                    "left_context": " ".join(q["left_context"]).strip(),
                    "mention": text_mention,
                    "right_context": " ".join(q["right_context"]).strip(),
                },  # dataset/task specific
            }
            data.append(datapoint)
    return data


class EntityLinkingDataset(Dataset):
    def __init__(
        self,
        name,
        input_file,
        output_file,
        id_filter_positive,
        id_filter_negative,
        max_chunks,
    ):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file
        self.ks = KnowledgeSource()
        self.id_filter_positive = id_filter_positive
        self.id_filter_negative = id_filter_negative
        self.max_chunks = max_chunks

    def get_chunks(self, num_chunks):

        data = []
        with open(self.input_file, "r") as fin:
            data = fin.readlines()

        # a single chunk for entity linking
        return [data]

    def process_chunk(self, lines, ks, chunk_id=-1):

        kilt_records = []

        # left context so far in the document
        left_context = []

        # working datapoints for the document
        document_questions = []

        # is the entity open
        open_entity = False

        # question id in the document
        question_i = 0

        for line in tqdm(lines):

            if "-DOCSTART-" in line:
                # new document is starting

                doc_id = line.split("(")[-1][:-2]

                # END DOCUMENT

                # check end of entity
                if open_entity:
                    open_entity = False

                """
                #DEBUG
                for q in document_questions:
                    pp.pprint(q)
                    input("...")
                """

                # add sentence_questions to kilt_records
                kilt_records.extend(
                    convert_to_KILT_format(
                        document_questions,
                        self.ks,
                        self.id_filter_positive,
                        self.id_filter_negative,
                    )
                )

                # reset
                left_context = []
                document_questions = []
                question_i = 0

            else:
                split = line.split("\t")
                token = split[0].strip()

                if len(split) >= 5:
                    B_I = split[1]
                    mention = split[2]
                    #  YAGO2_entity = split[3]
                    Wikipedia_URL = split[4]
                    Wikipedia_ID = split[5]
                    # Freee_base_id = split[6]

                    if B_I == "I":
                        pass

                    elif B_I == "B":

                        q = {
                            "id": "{}:{}".format(doc_id, question_i),
                            "mention": mention,
                            "Wikipedia_URL": Wikipedia_URL,
                            "Wikipedia_ID": Wikipedia_ID,
                            "left_context": left_context.copy(),
                            "right_context": [],
                        }
                        document_questions.append(q)
                        open_entity = True
                        question_i += 1

                    else:
                        print("Invalid B_I {}", format(B_I))
                        sys.exit(-1)

                    # print(token,B_I,mention,Wikipedia_URL,Wikipedia_ID)
                else:
                    if open_entity:
                        open_entity = False

                left_context.append(token)

                for q in document_questions[:-1]:
                    q["right_context"].append(token)

                if len(document_questions) > 0 and not open_entity:
                    document_questions[-1]["right_context"].append(token)

        # FINAL SENTENCE
        if open_entity:
            open_entity = False

        # add sentence_questions to kilt_records
        kilt_records.extend(
            convert_to_KILT_format(
                document_questions,
                self.ks,
                self.id_filter_positive,
                self.id_filter_negative,
            )
        )

        return kilt_records, []  # no metadata

    def postprocess_metadata(self, metadata):
        pass
