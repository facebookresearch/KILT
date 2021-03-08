# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import sys
import multiprocessing

from multiprocessing.pool import ThreadPool

from kilt.knowledge_source import KnowledgeSource


def run_thread(args):
    dataset = args["dataset"]
    return dataset.process_chunk(args["chunk"], args["ks"], args["id"])


def map_dataset(dataset):
    print("Processing {} dataset.".format(dataset.name))
    ks = KnowledgeSource()

    num_threads = (
        min(dataset.max_chunks, int(multiprocessing.cpu_count()))
        if dataset.max_chunks and dataset.max_chunks > 0
        else int(multiprocessing.cpu_count())
    )
    print("num_threads", num_threads)
    pool = ThreadPool(num_threads)
    chunks = dataset.get_chunks(num_threads)
    results = pool.map(
        run_thread,
        [
            {"id": id, "chunk": chunk, "ks": ks, "dataset": dataset}
            for id, chunk in enumerate(chunks)
        ],
    )

    kilt_data = []
    metadata = []
    for x in results:
        kd, meta = x
        kilt_data.extend(kd)
        metadata.append(meta)

    pool.terminate()
    pool.join()

    dataset.postprocess_metadata(metadata)

    with open(dataset.output_file, "w+") as outfile:
        for idx, data in enumerate(kilt_data):
            print(round(idx * 100 / len(kilt_data), 2), "%", end="\r")
            sys.stdout.flush()
            json.dump(data, outfile)
            outfile.write("\n")
