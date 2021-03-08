# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
import os

import spacy
import pprint

import kilt.kilt_utils as utils
from kilt.datasets.base_dataset import Dataset
from kilt.datasets.hotpotqa_ks import load_ks


class HotpotQADataset(Dataset):
    def __init__(
        self,
        name,
        input_file,
        output_file,
        log_file,
        ks_directory,
        get_only_original_evidence,
        max_chunks=None,
        debug=False,
    ):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file
        self.hotpotqa_ks = load_ks(ks_directory, verbose=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.max_chunks = max_chunks
        self.debug = debug
        self.get_only_original_evidence = get_only_original_evidence

    def get_chunks(self, num_chunks):
        all_data = []
        with open(self.input_file, "r") as fin:
            lines = fin.readlines()
            assert len(lines) == 1
            line = lines[0]
            all_data = json.loads(line)

        n = len(all_data)
        print("{} examples in the dataset".format(n))
        return utils.chunk_it(all_data, num_chunks)

    def process_chunk(self, chunk, ks, chunk_id=-1):

        missing_pages = 0.0
        exact_match = 0.0
        fuzzy_match = 0.0
        n = len(chunk)
        kilt_data = []
        metadata = []
        for idx, datapoint in enumerate(chunk):
            print(
                "t: {}, p: {:.2f} %, mp: {:.1f}, exact: {:.1f}, fuzzy: {:.1f}".format(
                    chunk_id,
                    round(idx * 100 / n, 2),
                    missing_pages,
                    exact_match,
                    fuzzy_match,
                ),
                end="\r",
            )
            sys.stdout.flush()

            kilt_record = {
                # original data point id if available otherwise unique id
                "id": datapoint["_id"],
                # question / claim / sentence
                "input": datapoint["question"],
                # dataset/task specific
                "meta": {"level": datapoint["level"], "type": datapoint["type"],},
            }
            kilt_record_provenance = []

            local_missing_page = False
            local_exact_match = True
            for evidence in datapoint["supporting_facts"]:
                title = evidence[0]
                sent_id = evidence[1]
                text = ""
                try:
                    text = self.hotpotqa_ks[title]["text"][sent_id]
                except IndexError as e:
                    print(
                        "\nIndexError: {}\ntitle:{}\nsent_id:{}\n".format(
                            e, title, sent_id
                        )
                    )

                if self.get_only_original_evidence:
                    kilt_record_provenance.append(
                        {"text": text, "title": title, "sent_id": sent_id}
                    )

                else:
                    pages = ks.get_pages_by_title(title)
                    if len(pages) == 0:
                        local_missing_page = True
                        break

                    bleu = -1
                    paragraph_id = -1
                    start_character = -1
                    end_character = -1
                    for page in pages:
                        # it is unlikely, but there could be multiple pages for a title (e.g., disambiguation)
                        if text and len(text) > 0:
                            (
                                local_paragraph_id,
                                local_start_character,
                                local_end_character,
                                local_bleu,
                            ) = utils.match_answer(
                                text, page, nlp=self.nlp, debug=False
                            )

                            if local_bleu > bleu:
                                paragraph_id = local_paragraph_id
                                start_character = local_start_character
                                end_character = local_end_character
                                bleu = local_bleu

                    if bleu != 1.0:
                        local_exact_match = False

                    kilt_record_provenance.append(
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
                        }
                    )

            if local_missing_page:
                missing_pages += 1
                continue
            if local_exact_match:
                exact_match += 1
            else:
                fuzzy_match += 1

            kilt_record["output"] = [
                {"answer": datapoint["answer"], "provenance": kilt_record_provenance}
            ]
            kilt_data.append(kilt_record)

            if self.debug:
                pp = pprint.PrettyPrinter(indent=4)
                print("original datapoint:")
                pp.pprint(datapoint)
                input("...")
                print("kilt record:")
                pp.pprint(kilt_record)
                input("...")

        metadata = [missing_pages, exact_match, fuzzy_match]
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
