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


class NaturalQuestionsDataset(Dataset):
    def __init__(self, name, input_file, output_file, log_file):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file
        self.nlp = spacy.load("en_core_web_sm")

    def get_chunks(self, num_chunks):
        all_data = []
        with open(self.input_file, "r") as infile:
            for line in infile:
                data = json.loads(line)
                all_data.append(data)

        n = len(all_data)
        print("{} examples in the dataset".format(n))
        return utils.chunk_it(all_data, num_chunks)

    def process_chunk(self, chunk, ks, chunk_id=-1):
        missing_pages = 0.0
        short_exact_match = 0.0
        short_fuzzy_match = 0.0
        n = len(chunk)
        kilt_data = []
        metadata = []

        for idx, datapoint in enumerate(chunk):

            # from standard to simplified format
            if "document_text" not in datapoint:
                # wget https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/text_utils.py
                from text_utils import simplify_nq_example
                datapoint = simplify_nq_example(datapoint)

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

            url = datapoint["document_url"]
            page = ks.get_page_from_url(url)

            if not page:
                print("ERROR, not page!")
                missing_pages += 1
            else:
                # get and validate annotations
                annotations = datapoint["annotations"]

                kilt_record = {
                    # original data point id if available otherwise unique id
                    "id": datapoint["example_id"],
                    # question / claim / sentence
                    "input": datapoint["question_text"],
                }

                kilt_record_output = []
                local_sem = 0.0
                local_sfm = 0.0

                for annotation in annotations:

                    if "short_answers" in annotation:
                        short_answers = annotation["short_answers"]

                        # scan all possible short answers
                        for answer_index in range(len(short_answers)):
                            s = short_answers[answer_index]["start_token"]
                            e = short_answers[answer_index]["end_token"]
                            short_answer = datapoint["document_text"].split()[s:e]
                            answer_span = " ".join(short_answer).strip()

                            (
                                paragraph_id,
                                start_character,
                                end_character,
                                bleu,
                            ) = utils.match_answer(
                                answer_span, page, nlp=self.nlp, debug=False
                            )

                            kilt_record_output.append(
                                {
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
                                            "meta": {  # dataset/task specific
                                                "yes_no_answer": annotations[0][
                                                    "yes_no_answer"
                                                ],
                                                "annotation_id": annotations[0][
                                                    "annotation_id"
                                                ],
                                            },
                                        }
                                    ],
                                }
                            )

                            if bleu == 1:
                                local_sem += 1
                            elif bleu < 1 and bleu >= 0:
                                local_sfm += 1
                            else:
                                print("ERROR: invalid bleu: {}".format(bleu))
                                sys.exit(-1)

                    if "long_answer" in annotation:

                        long_answer = annotation["long_answer"]

                        s = long_answer["start_token"]
                        e = long_answer["end_token"]
                        long_answer = datapoint["document_text"].split()[s:e]
                        answer_span = " ".join(long_answer).strip()

                        (
                            paragraph_id,
                            start_character,
                            end_character,
                            bleu,
                        ) = utils.match_answer(
                            answer_span, page, nlp=self.nlp, debug=False
                        )

                        kilt_record_output.append(
                            {
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
                                        "meta": {  # dataset/task specific
                                            "yes_no_answer": annotations[0][
                                                "yes_no_answer"
                                            ],
                                            "annotation_id": annotations[0][
                                                "annotation_id"
                                            ],
                                        },
                                    }
                                ],
                            }
                        )

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

                # average by answers per single question
                # if len(short_answers) > 0:
                #     short_exact_match += local_sem / len(short_answers)
                #     short_fuzzy_match += local_sfm / len(short_answers)

        metadata = [missing_pages, short_exact_match, short_fuzzy_match]
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