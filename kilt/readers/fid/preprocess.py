# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import sys
from tqdm.auto import tqdm


def convert_kilt(inputpath, outputpath):
    data = []
    inputdata = open(inputpath, "r")
    for example in tqdm(inputdata):
        d = {}
        ex = json.loads(example)
        d["question"] = ex["input"]
        answers = set()
        for a in ex["output"]:
            if "answer" in a:
                answers.add(a["answer"])
        d["answers"] = list(answers)
        d["id"] = ex["id"]
        passages = []
        for c in ex["output"][0]["provenance"]:
            p = {"text": c["text"], "title": ""}
            if "wikipedia_title" in c:
                p["title"] = c["wikipedia_title"]
            if "wikipedia_id" in c:
                p["wikipedia_id"] = c["wikipedia_id"]
            passages.append(p)
        d["ctxs"] = passages
        data.append(d)
    with open(outputpath, "w") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    convert_kilt(inputpath, outputpath)
