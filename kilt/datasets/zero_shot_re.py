# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import spacy
import uuid

import kilt.kilt_utils as utils
from kilt.datasets.base_dataset import Dataset


class ZeroShotREDataset(Dataset):
    def __init__(self, name, input_file, output_file, max_chunks):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file
        self.max_chunks = max_chunks
        self.nlp = spacy.load("en_core_web_sm")

    def get_uuid(self):
        return str(uuid.uuid4())

    def map_datapoint(
        self,
        wikidata_relation,
        question_template,
        wikipedia_title,
        sentence,
        answer_spans,
        ks,
        entry_id,
    ):
        kilt_entry = {}
        kilt_entry["id"] = entry_id
        kilt_entry["input"] = question_template.replace("XXX", wikipedia_title).replace(
            " auther", " author"  # to fix typo in templates
        )
        kilt_entry["output"] = []
        kilt_entry["meta"] = {
            "wikidata_relation": wikidata_relation,
            "question_template": question_template,
        }
        print("Getting wiki page for", wikipedia_title)
        pages = ks.get_pages_by_title(wikipedia_title)

        if len(pages) <= 0:
            kilt_entry["output"] = [
                {"answer": answer_span, "provenance": []}
                for answer_span in answer_spans
            ]
            return kilt_entry
        print("matching answer")
        # We take the first returned page from the list.
        paragraph_id, start_character, end_character, bleu = utils.match_answer(
            sentence, pages[0], nlp=self.nlp, debug=False
        )
        print("done matching answer")

        for answer_span in answer_spans:
            output = {"answer": answer_span, "provenance": []}
            output["provenance"].append(
                {
                    "wikipedia_id": pages[0]["wikipedia_id"],
                    "title": pages[0]["wikipedia_title"],
                    "start_paragraph_id": paragraph_id,
                    "start_character": start_character,
                    "end_paragraph_id": paragraph_id,
                    "end_character": end_character,
                    "bleu_score": bleu,
                    "meta": {},
                }
            )
            kilt_entry["output"].append(output)
        return kilt_entry

    def get_chunks(self, num_chunks):
        data = []
        with open(self.input_file, "r") as fin:
            data = fin.readlines()
        return utils.chunk_it(data, num_chunks)

    def process_chunk(self, chunk, ks, chunk_id):
        kilt_data = []
        missing_pages = 0
        negative_samples = 0
        for i, line in enumerate(chunk):
            print("Processed {} lines for chunk {}".format(i, chunk_id))
            print("Processing:", line)
            fields = line.strip().split("\t")
            # Leave out negative samples (samples where one can't infer the
            # answer from the provided sentence).
            if len(fields) <= 4:
                negative_samples += 1
                continue
            wikidata_relation, question_template, wikipedia_title, sentence = fields[
                0:4
            ]
            answer_spans = fields[4:]
            kilt_entry = self.map_datapoint(
                wikidata_relation,
                question_template,
                wikipedia_title,
                sentence,
                answer_spans,
                ks,
                self.get_uuid(),
            )
            if kilt_entry is None:
                missing_pages += 1
                continue
            kilt_data.append(kilt_entry)
        return kilt_data, [missing_pages, negative_samples]

    def postprocess_metadata(self, metadata):
        missing_pages = 0
        negative_samples = 0
        for m, n in metadata:
            missing_pages += m
            negative_samples += n
        print(
            "{} samples with missing pages, {} samples with no answer spans.".format(
                missing_pages, negative_samples
            )
        )
