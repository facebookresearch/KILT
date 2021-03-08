# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import spacy
import sys
import re
import kilt.kilt_utils as utils
from kilt.datasets.base_dataset import Dataset
from kilt import knowledge_source # remove later


class TriviaQADataset(Dataset):
    def __init__(self, name, input_file, output_file, log_file):
        super().__init__(name)
        self.input_file = "/private/home/angelafan/robocheckers/KIB/wikipedia-test.json"
        self.output_file = output_file
        self.log_file = log_file
        self.nlp = spacy.load("en_core_web_sm")

    def get_chunks(self, num_chunks):
        with open(self.input_file, "r", encoding='utf-8') as infile:
            all_data = json.load(infile)

        all_data = all_data['Data']
        n = len(all_data)
        print("{} examples in the dataset".format(n))
        return utils.chunk_it(all_data, num_chunks)

    def process_chunk(self, chunk, ks, chunk_id=-1):
        missing_pages = 0.0
        short_exact_match = 0.0
        short_fuzzy_match = 0.0
        n = len(chunk)
        kilt_data = []

        for idx, datapoint in enumerate(chunk):

            print(
                "t: {}, p: {:.2f} %, mp: {:.1f}, exact: {:.1f}, fuzzy: {:.1f}".format(
                    chunk_id,
                    round(idx * 100 / n, 2),
                    missing_pages,
                    short_exact_match,
                    short_fuzzy_match,
                ),
                end="\r",
            )
            sys.stdout.flush()

            # answer
            answers = datapoint["Answer"]["Aliases"]
            normalized_answers = datapoint["Answer"]["NormalizedAliases"]
            question = datapoint["Question"]
            wikipedia_pages = datapoint["EntityPages"]
            wiki_titles = [i["Title"] for i in wikipedia_pages]
            dataset_id = datapoint["QuestionId"]

            # group by question,
            for answer_index, answer in enumerate(answers):
                for title in wiki_titles:
                    page = ks.get_pages_by_title(title)
                    if not page:
                        missing_pages += 1 # metric will be inflated since its on each unfetchable page
                    else:
                        page = page[0]
                        kilt_record = {
                            # original data point id if available otherwise unique id
                            "id": dataset_id,
                            # question / claim / sentence
                            # dialogue history goes here
                            "input": question,
                        }

                        local_sem = 0.0
                        local_sfm = 0.0

                        answer_span = answer

                        (
                            paragraph_id,
                            start_character,
                            end_character,
                            bleu,
                        ) = utils.match_answer(
                            answer_span, page, nlp=self.nlp, debug=False
                        )

                        kilt_record_output = {
                            # answer in textual form
                            "answer": answer_span,
                            "provenance": [
                                # list of relevant WikipediaPages / Spans as provenance for the answer from the ks
                                {
                                    "wikipedia_id": page[
                                        "wikipedia_id"
                                    ],  # *mandatory* - ID Wikipedia Page
                                    "title": page[
                                        "wikipedia_title"
                                    ],  # *mandatory* - Title Wikipedia Page
                                    "start_paragraph_id": paragraph_id,  # start paragraph id with relevant info
                                    "start_character": start_character,
                                    "end_paragraph_id": paragraph_id,  # end paragraph id
                                    "end_character": end_character,
                                    "bleu_score": bleu,  # 1.0 when gold data is exactly matched, lower for fuzzy matches
                                    "normalized_aliases": normalized_answers
                                }
                            ],
                        }


                        if bleu == 1:
                            local_sem += 1
                        elif bleu < 1 and bleu >= 0:
                            local_sfm += 1
                        else:
                            print("ERROR: invalid bleu: {}".format(bleu))
                            sys.exit(-1)

                        # update kilt data
                        kilt_record["output"] = kilt_record_output
                        kilt_data.append(kilt_record)

        metadata = [missing_pages]
        return kilt_data, metadata

    def postprocess_metadata(self, metadata):
        missing_pages = 0.0
        short_exact_match = 0.0
        short_fuzzy_match = 0.0
        for met in metadata:
            if met == []:
                continue
            mp, sem, sfm = met
            missing_pages += mp
            short_exact_match += sem
            short_fuzzy_match += sfm

        print("Print stats")
        msg = "\n n: {:.1f}, missing pages: {:.1f}, short exact match: {:.1f}, short fuzzy match: {:.1f}".format(
            0, missing_pages, short_exact_match, short_fuzzy_match
        )
        print(msg)

        f = open(self.log_file, "w+")
        f.write(msg)
        f.close()
