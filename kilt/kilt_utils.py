# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import nltk
import json
import os
import logging
import sys
import time
import string
import random


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_punc(lower(s))


def validate_datapoint(datapoint, logger):

    # input is a string
    if not isinstance(datapoint["input"], str):
        if logger:
            logger.warning(
                "[{}] input is not a string {}".format(
                    datapoint["id"], datapoint["input"]
                )
            )
        return False

    # output is not empty
    if "output" in datapoint:
        if len(datapoint["output"]) == 0:
            if logger:
                logger.warning("[{}] empty output".format(datapoint["id"]))
            return False

        for output in datapoint["output"]:
            # answer is a string
            if "answer" in output:
                if not isinstance(output["answer"], str):
                    if logger:
                        logger.warning(
                            "[{}] answer is not a string {}".format(
                                datapoint["id"], output["answer"]
                            )
                        )
                    return False

            # provenance is not empty
            # if len(output["provenance"]) == 0:
            #    if logger:
            #        logger.warning("[{}] empty provenance".format(datapoint["id"]))
            #    return False

            if "provenance" in output:
                for provenance in output["provenance"]:
                    # wikipedia_id is provided
                    if not isinstance(provenance["wikipedia_id"], str):
                        if logger:
                            logger.warning(
                                "[{}] wikipedia_id is not a string {}".format(
                                    datapoint["id"], provenance["wikipedia_id"]
                                )
                            )
                        return False

                    # title is provided
                    if not isinstance(provenance["title"], str):
                        if logger:
                            logger.warning(
                                "[{}] title is not a string {}".format(
                                    datapoint["id"], provenance["title"]
                                )
                            )
                        return False

    return True


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def store_data(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")


def get_bleu(candidate_tokens, gold_tokens):

    candidate_tokens = [x for x in candidate_tokens if len(x.strip()) > 0]
    gold_tokens = [x for x in gold_tokens if len(x.strip()) > 0]

    # The default BLEU calculates a score for up to
    # 4-grams using uniform weights (this is called BLEU-4)
    weights = (0.25, 0.25, 0.25, 0.25)

    if len(gold_tokens) < 4:
        # lower order ngrams
        weights = [1.0 / len(gold_tokens) for _ in range(len(gold_tokens))]

    BLEUscore = nltk.translate.bleu_score.sentence_bleu(
        [candidate_tokens], gold_tokens, weights=weights
    )
    return BLEUscore


# split a list in num parts evenly
def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num  # 0 <= diff < num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def init_logging(base_logdir, modelname, logger=None):

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_directory = "{}/{}/".format(base_logdir, modelname)

    if logger == None:
        logger = logging.getLogger("KILT")

        logger.setLevel(logging.DEBUG)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    else:
        # remove previous file handler
        logger.handlers.pop()

    os.makedirs(log_directory, exist_ok=True)

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.propagate = False
    logger.info("logging in {}".format(log_directory))
    return logger


def create_logdir_with_timestamp(base_logdir):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, timestr, random.randint(0, 1000))
    os.makedirs(log_directory)
    return log_directory
