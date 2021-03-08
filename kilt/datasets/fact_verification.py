# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import spacy
import sys
import unicodedata


import pprint

pp = pprint.PrettyPrinter(indent=4)

import kilt.kilt_utils as utils
from kilt.datasets.base_dataset import Dataset


class FactVerificationDataset(Dataset):
    def __init__(
        self, name, claims_input_file, evidence_directory_path, output_file, log_file
    ):
        super().__init__(name)
        self.claims_input_file = claims_input_file
        self.evidence_directory_path = evidence_directory_path
        self.output_file = output_file
        self.log_file = log_file
        self.nlp = spacy.load("en_core_web_sm")

    def _normalize(self, text):
        replacements = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
            "-COLON-": ":",
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        return text

    def get_chunks(self, num_chunks):

        # Read claims, create a set of wiki pages to
        # find the evidence sentences in
        page_to_evidence_sents = {}

        with open(self.claims_input_file, "r") as infile:
            for line in infile:
                claim = json.loads(line)

                if "verifiable" in claim and claim["verifiable"] == "NOT VERIFIABLE":
                    continue

                evidence_sets = claim["evidence"]
                for evidence_set in evidence_sets:

                    for evidence in evidence_set:
                        if evidence[2]:
                            page_id = unicodedata.normalize("NFKD", evidence[2])
                        else:
                            #  those can be filtered out/ignored. They’re an artefact of merging some of the duplicates where annotators disagreed over the label.
                            break

                        sent_id = int(evidence[3])

                        if page_id not in page_to_evidence_sents:
                            page_to_evidence_sents[page_id] = {}

                        page_to_evidence_sents[page_id][sent_id] = None

        for idx in range(1, 110):
            filename = self.evidence_directory_path + f"/wiki-{idx:03}.jsonl"
            print(f"processing filename {filename}")
            with open(filename, "r") as fin:
                for line in fin:
                    wiki_page = json.loads(line.strip())
                    page_id = wiki_page["id"]
                    if page_id not in page_to_evidence_sents:
                        continue
                    lines = wiki_page["lines"].split("\n")
                    sentences = []
                    for l in lines:
                        line_fields = l.split("\t")
                        # skip empty sentences
                        if len(line_fields) < 2 or line_fields[1] == "":
                            continue
                        # skip sentences where first element is not number
                        if not line_fields[0].isdigit():
                            continue

                        sent_text = line_fields[1]

                        # there is no id, so the new line character is
                        # likely a formatting error, will ignore and
                        # append the normalized text to the previous
                        # sentence.
                        if line_fields[0] == "":
                            sentences[-1]["text"] += " " + sent_text
                        else:
                            sentences.append(
                                {
                                    "id": line_fields[0],
                                    "text": sent_text,
                                }
                            )

                    for sentence in sentences:
                        sent_id = int(sentence["id"])
                        sent_text = sentence["text"]
                        if sent_id in page_to_evidence_sents[page_id]:
                            page_to_evidence_sents[page_id][sent_id] = sent_text

        data = []
        for page_id in page_to_evidence_sents:
            for sent_id in page_to_evidence_sents[page_id]:
                sent_text = page_to_evidence_sents[page_id][sent_id]
                data.append(
                    {
                        "page_id": page_id,
                        "sent_id": sent_id,
                        "text": sent_text,
                    }
                )

        n = len(data)
        print("{} examples in the dataset".format(n))
        return utils.chunk_it(data, num_chunks)

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

            page_id = datapoint["page_id"]
            sent_id = datapoint["sent_id"]
            text = datapoint["text"]

            if not text or text == None or len(text) == 0:
                continue

            url = "https://en.wikipedia.org/wiki/" + self._normalize(
                datapoint["page_id"]
            )
            page = ks.get_page_from_url(url)
            if not page:
                missing_pages += 1
            else:
                # get and validate evidence sentence

                local_sem = 0.0
                local_sfm = 0.0

                kilt_record = {
                    # original data point id if available otherwise unique id
                    "page_id": page_id,
                    "sentence_id": sent_id,
                    "evidence_text": text,
                }

                kilt_record_output = []

                paragraph_id, start_character, end_character, bleu = utils.match_answer(
                    text, page, nlp=self.nlp, debug=False
                )

                kilt_record_output.append(
                    {
                        # answer in textual form
                        "answer": text,
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
                                    "fever_page_id": page_id,
                                    "fever_sentence_id": sent_id,
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

                exact_match += local_sem  # / len(short_answers)
                fuzzy_match += local_sfm  # / len(short_answers)

                metadata = [missing_pages, exact_match, fuzzy_match]

        return kilt_data, metadata

    def postprocess_metadata(self, metadata):
        missing_pages = 0.0
        exact_match = 0.0
        fuzzy_match = 0.0
        for met in metadata:
            if met == []:
                continue
            mp, sem, sfm = met
            missing_pages += mp
            exact_match += sem
            fuzzy_match += sfm

        print("Print stats")
        msg = "\n n: {:.1f}, missing pages: {:.1f}, exact match: {:.1f}, fuzzy match: {:.1f}".format(
            0, missing_pages, exact_match, fuzzy_match
        )
        print(msg)

        f = open(self.log_file, "w+")
        f.write(msg)
        f.close()
