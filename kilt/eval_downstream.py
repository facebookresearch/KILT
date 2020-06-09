import argparse
import json
import jsonlines
import pprint
import re
import string
import sys

from collections import Counter

import kilt.evaluation_metrics as retrieval_metrics
from kilt import kilt_utils


def exact_match(guess_dataset, gold_dataset):
    """
    Calculate exact match score between two datasets.

    Args:
        guess_data_set (list of KILT json data): The KILT instances to compare against the gold data
        gold_data_set (list of KILT json data): The KILT instances to compare against the guess data.
    """
    total_count = 0
    total_matches = 0
    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = set(item["answer"] for item in gold_item["output"])
        guess_candidate_answers = set(item["answer"] for item in guess_item["output"])
        if all(x in gold_candidate_answers for x in guess_candidate_answers):
            total_matches += 1
    return total_matches / total_count


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def qa_exact_match(guess_dataset, gold_dataset):
    total = 0
    total_em = 0
    for guess, gold in zip(guess_dataset, gold_dataset):
        if len(gold["output"]) == 0:
            raise Exception("bad gold")
        ground_truths = [item["answer"] for item in gold["output"]]
        total_em += metric_max_over_ground_truths(
            exact_match_score_qa, guess["output"][0]["answer"], ground_truths
        )
        total += 1
    return total_em / total


def kilt_qa_exact_match(guess_dataset, gold_dataset):
    total = 0
    total_em = 0
    for guess, gold in zip(guess_dataset, gold_dataset):
        ranking_metrics = retrieval_metrics.get_ranking_metrics(guess, gold)
        if ranking_metrics["Rprec"] == 1:
            ground_truths = [item["answer"] for item in gold["output"]]
            total_em += metric_max_over_ground_truths(
                exact_match_score_qa, guess["output"][0]["answer"], ground_truths
            )
        total += 1
    return total_em / total


def qa_f1(guess_dataset, gold_dataset):
    total = 0
    total_f1 = 0
    for guess, gold in zip(guess_dataset, gold_dataset):
        if len(gold["output"]) == 0:
            raise Exception("bad gold")
        ground_truths = [item["answer"] for item in gold["output"]]
        total_f1 += metric_max_over_ground_truths(
            f1_score, guess["output"][0]["answer"], ground_truths
        )
        total += 1
    return total_f1 / total


def kilt_qa_f1(guess_dataset, gold_dataset):
    total = 0
    total_em = 0
    for guess, gold in zip(guess_dataset, gold_dataset):
        ranking_metrics = retrieval_metrics.get_ranking_metrics(guess, gold)
        if ranking_metrics["Rprec"] == 1:
            ground_truths = [item["answer"] for item in gold["output"]]
            total_em += metric_max_over_ground_truths(
                f1_score, guess["output"][0]["answer"], ground_truths
            )
        total += 1
    return total_em / total


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    if not prediction:
        # print("WARNING: Null prediction")
        return 0.0
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score_qa(prediction, ground_truth):
    if not prediction:
        #  print("WARNING: Null prediction")
        return 0.0
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    for gold, guess in zip(gold_records, guess_records):
        assert gold["id"] == guess["id"], "Items must have same order with same IDs"

    return {
        "em": exact_match(guess_records, gold_records),
        "em_qa": qa_exact_match(guess_records, gold_records),
        "f1_qa": qa_f1(guess_records, gold_records),
    }


def calculate_kilt_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    for gold, guess in zip(gold_records, guess_records):
        assert gold["id"] == guess["id"], "Items must have same order with same IDs"

    return {
        "kilt_em": kilt_qa_exact_match(guess_records, gold_records),
        "kilt_f1": kilt_qa_f1(guess_records, gold_records),
    }


def validate_input(gold_records, guess_records):

    if len(gold_records) != len(guess_records):
        print(
            "WARNING: DIFFERENT SIZE gold: {} guess: {}".format(
                len(gold_records), len(guess_records)
            )
        )
        sys.exit(-1)

    # align order
    gold_ids = []
    for gold in gold_records:
        assert gold["id"] not in gold_ids, "Gold IDs should be unique"
        gold_ids.append(gold["id"])

    id2guess_record = {}
    for guess in guess_records:
        id2guess_record[guess["id"]] = guess

    guess_records = []
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])

    temp_gold_records = []
    for gold in gold_records:
        if gold["id"] in id2guess_record:
            temp_gold_records.append(gold)
    gold_records = temp_gold_records

    return gold_records, guess_records


def evaluate(gold, guess):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = kilt_utils.load_data(gold)
    guess_records = kilt_utils.load_data(guess)

    # TODO 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. retrieval performance
    retrieval_result = retrieval_metrics.compute(gold_records, guess_records)
    print("retrieval_result:")
    pp.pprint(retrieval_result)

    # 2. end2end results
    e2e_result = calculate_metrics(gold_records, guess_records)
    print("\ne2e_result:")
    pp.pprint(e2e_result)

    # 2. KILT Score
    kilt_result = calculate_kilt_metrics(gold_records, guess_records)
    print("\nkilt_result:")
    pp.pprint(kilt_result)

    return (
        round(retrieval_result["Rprec"] * 100, 2),
        round(e2e_result["em_qa"] * 100, 2),
        round(e2e_result["f1_qa"] * 100, 2),
        round(kilt_result["kilt_em"] * 100, 2),
        round(kilt_result["kilt_f1"] * 100, 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")

    args = parser.parse_args()
    evaluate(args.gold, args.guess)
