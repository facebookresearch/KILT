# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys
import requests
import tarfile
import os
import json

from tqdm.auto import tqdm

from kilt import kilt_utils

url = "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
tar_filename = "triviaqa-rc.tar.gz"
trivia_path = "triviaqa-rc/"
members = [
    "qa/wikipedia-train.json",
    "qa/wikipedia-dev.json",
    "qa/wikipedia-test-without-answers.json",
]
base = "data/"
input_files = [
    base + "triviaqa-train_id-kilt.jsonl",
    base + "triviaqa-dev_id-kilt.jsonl",
    base + "triviaqa-test_id_without_answers-kilt.jsonl",
]
output_files = [
    base + "triviaqa-train-kilt.jsonl",
    base + "triviaqa-dev-kilt.jsonl",
    base + "triviaqa-test_without_answers-kilt.jsonl",
]


def decompress(tar_file, path, members=None):
    """
    Extracts `tar_file` and puts the `members` to `path`.
    If members is None, all members on `tar_file` will be extracted.
    """
    tar = tarfile.open(tar_file, mode="r:gz")
    if members is None:
        members = tar.getmembers()
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        tar.extract(member, path=path)
        # set the progress description of the progress bar
        progress.set_description(f"Extracting {str(member)}")
    # or use this
    # tar.extractall(members=members, path=path)
    # close the file
    tar.close()


print("1. download TriviaQA original tar.gz file")
# Streaming, so we can iterate over the response.
r = requests.get(url, stream=True)
# Total size in bytes.
total_size = int(r.headers.get("content-length", 0))
block_size = 1024  # 1 Kibibyte
t = tqdm(total=total_size, unit="iB", unit_scale=True)
with open(tar_filename, "wb") as f:
    for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)
t.close()
if total_size != 0 and t.n != total_size:
    print("ERROR, something went wrong")


print("2. extract tar.gz file")
decompress(tar_filename, trivia_path, members=members)

print("3. remove tar.gz file")
os.remove(tar_filename)

print("4. getting original questions")
id2input = {}
for member in members:
    print(member)
    filename = trivia_path + member
    with open(filename, "r") as fin:
        data = json.load(fin)
        for element in data["Data"]:
            e_id = element["QuestionId"]
            e_input = element["Question"]
            assert e_id not in id2input
            id2input[e_id] = e_input
    os.remove(filename)

print("5. remove original TriviaQA data")
os.rmdir(trivia_path + "qa/")
os.rmdir(trivia_path)

print("6. update kilt files")
for in_file, out_file in zip(input_files, output_files):
    data = kilt_utils.load_data(in_file)
    for element in data:
        element["input"] = id2input[element["id"]]
    kilt_utils.store_data(out_file, data)
    os.remove(in_file)
