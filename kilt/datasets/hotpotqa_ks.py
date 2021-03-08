# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from kilt.kilt_utils import chunk_it
import bz2
import json

STEPS = 10


def run_thread(arguments):
    thread_id = arguments["id"]
    filenames = arguments["filenames"]
    verbose = arguments["verbose"]

    output_dict = {}

    steps = int(len(filenames) / STEPS)

    for file_id, filename in enumerate(filenames):

        if verbose:
            try:
                if file_id % steps == 0:
                    percentage = file_id * 100 / len(filenames)
                    print(
                        "t{} [{}/{}] {:.2f}%".format(
                            thread_id, file_id, len(filenames), percentage
                        ),
                        flush=True,
                    )
            except:
                pass
        with bz2.open(
            filename,
            mode="r",
            compresslevel=9,
            encoding=None,
            errors=None,
            newline=None,
        ) as f:
            for line in f:
                data = json.loads(line)
                output_dict[data["title"]] = data

    return output_dict


def load_ks(ks_directory, verbose=False):
    NUM_TREADS = int(multiprocessing.cpu_count())

    if verbose:
        print(f"loading hotpotqa knowledge source with {NUM_TREADS} threads")
    pool = ThreadPool(NUM_TREADS)

    filenames = []
    directories = [
        os.path.join(ks_directory, o)
        for o in os.listdir(ks_directory)
        if os.path.isdir(os.path.join(ks_directory, o))
    ]
    for directory in directories:
        onlyfiles = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        for filetto in onlyfiles:
            filename = "{}/{}".format(directory, filetto)
            filenames.append(filename)

    arguments = [
        {"id": i, "filenames": chunk, "verbose": verbose}
        for i, chunk in enumerate(chunk_it(filenames, NUM_TREADS))
    ]

    results = pool.map(run_thread, arguments)
    output_dict = {}
    for x in results:
        output_dict.update(x)
    pool.terminate()
    pool.join()

    return output_dict
