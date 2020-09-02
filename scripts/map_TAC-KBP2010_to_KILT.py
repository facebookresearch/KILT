# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from tqdm.auto import tqdm
import pickle
import argparse

from kilt.knowledge_source import KnowledgeSource


def write_output(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--train",
    dest="train_mentions_filename",
    type=str,
    default="data/tac_kbp_2010/train.jsonl",
    help="train file TAC-KBP2010",
)

parser.add_argument(
    "--test",
    dest="test_mentions_filename",
    type=str,
    default="data/tac_kbp_2010/test.jsonl",
    help="train file TAC-KBP2010",
)

parser.add_argument(
    "--entities",
    dest="test_entities_filename",
    type=str,
    default="data/tac_kbp_2010/tac_kbp_ref_know_base/entity.jsonl",
    help="knowledge source file TAC-KBP2010",
)

parser.add_argument(
    "--out_test",
    dest="out_test",
    type=str,
    default="data/tac_kbp_2010/tackbp2010-test-kilt.jsonl",
    help="output file for TAC-KBP2010 test in KILT format",
)

parser.add_argument(
    "--out_train",
    dest="out_train",
    type=str,
    default="data/tac_kbp_2010/tackbp2010-train-kilt.jsonl",
    help="output file for TAC-KBP2010 train in KILT format",
)

args = parser.parse_args()

ent_start_token = "[START_ENT]"
ent_end_token = "[END_ENT]"
ks = KnowledgeSource()
kb2id = {}

manual_labels_correspondance = {
    "E0431500": 19457,  # Myanmar
    "E0633385": 109495,  # Key West
    "E0277953": 8725021,  # Aarti Agarwal
    "E0526355": 30875653,  # Bob Casey Jr.
    "E0508649": 41709552,  # American Eagle (airline brand)
    "E0504008": 504790,  # New York Daily News
    "E0398776": 99689,  # National Express
    "E0343020": 402982,  # Reliance Industries Limited
    "E0131583": 77825,  # TNT (American TV network)
    "E0586856": 12710981,  # List of Dirty Sexy Money characters
    "E0439840": 1114732,  # Palestine (region)
    "E0655951": 607797,  # Miami Herald
    "E0681609": 7761399,  # Chad Johnson
    "E0233160": 27169389,  # Ronald Reagan UCLA Medical Center
    "E0465278": 7554772,  # Randalls
    "E0435757": 2118244,  # Bago, Myanmar
    "E0194326": 14141082,  # Belmond Limited
    "E0029703": 30858216,  # Aaj News
    "E0071026": 27885464,  # Public Security Police Force of Macau
    "E0513036": 14331070,  # Senvion
    "E03912200": None,  # Nepal Cable Television Association
    "E0436955": None,  # PAS
}

labels = {}
with open(args.train_mentions_filename, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        data = json.loads(line)
        label_id = str(data["label_id"]).strip()
        if label_id not in labels:
            labels[label_id] = False

with open(args.test_mentions_filename, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        data = json.loads(line)
        label_id = str(data["label_id"]).strip()
        if label_id not in labels:
            labels[label_id] = False

print("labels:", len(labels))
missing_pages = 0
with open(args.test_entities_filename, "r") as fin:
    lines = fin.readlines()
    for line in tqdm(lines):
        entity = json.loads(line)
        title = entity["title"]
        kb_idx = str(entity["kb_idx"]).strip()

        if kb_idx in labels:
            labels[kb_idx] = True
            title = title.replace("&amp;", "&")
            page = ks.get_page_by_title(title)
            if page:
                kb2id[kb_idx] = page["wikipedia_id"]
            else:
                missing_pages += 1

c = 0
for label, found in labels.items():
    if not found:
        if (
            label in manual_labels_correspondance
            and manual_labels_correspondance[label]
        ):
            kb2id[label] = manual_labels_correspondance[label]
        else:
            c += 1
print(f"missing {c}/{len(labels)} labels in ks")

for idx, filename in enumerate(
    [args.test_mentions_filename, args.train_mentions_filename]
):
    kilt_records = []
    missing = 0
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            label_id = str(data["label_id"]).strip()
            if label_id in kb2id:
                wikipedia_id = kb2id[label_id]
                page = ks.get_page_by_id(wikipedia_id)

                input_text = (
                    str(data["context_left"]).strip()
                    + " "
                    + ent_start_token
                    + " "
                    + str(data["mention"]).strip()
                    + " "
                    + ent_end_token
                    + " "
                    + str(data["context_right"]).strip()
                )

                # rename
                data["left_context"] = data.pop("context_left")
                data["right_context"] = data.pop("context_right")

                kilt_records.append(
                    {
                        "id": data["query_id"],
                        "input": input_text,
                        "output": [
                            {
                                "answer": page["wikipedia_title"],
                                "provenance": [
                                    {
                                        "wikipedia_id": wikipedia_id,
                                        "title": page["wikipedia_title"],
                                    }
                                ],
                            }
                        ],
                        "meta": data,
                    }
                )
            else:
                missing += 1

    if idx == 1:
        print("missing {}/{} points in train".format(missing, len(lines)))
        write_output(args.out_train, kilt_records)
    elif idx == 0:
        print("missing {}/{} points in test".format(missing, len(lines)))
        write_output(args.out_test, kilt_records)
    else:
        print("ERROR")
