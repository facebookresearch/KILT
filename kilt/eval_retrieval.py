# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import pprint
from collections import defaultdict, OrderedDict

from kilt import kilt_utils
from kilt import eval_downstream


def _remove_duplicates(obj):
    obj_tmp = []
    for o in obj:
        if o not in obj_tmp:
            obj_tmp.append(o)
    return obj_tmp


def _get_ids_list(datapoint, rank_keys, verbose=False):
    # collect all gold ids
    ids_list = []
    for output in datapoint["output"]:
        current_ids_list = []
        if "provenance" in output:
            for provenance in output["provenance"]:
                if any(rank_key not in provenance for rank_key in rank_keys):
                    missing = set(rank_keys) - set(
                        list(provenance.keys())
                    ).intersection(set(rank_keys))
                    if verbose:
                        print(
                            f"WARNING: missing key(s) {missing} in provenance, unable to compute retrieval for those."
                        )
                else:
                    current_ids_list.append(
                        "+".join(
                            [
                                str(provenance[rank_key]).strip()
                                for rank_key in rank_keys
                            ]
                        )
                    )
        ids_list.append(_remove_duplicates(current_ids_list))  # remove duplicates

    # consider only unique ids
    return ids_list


def compute_rank_for_item(guess_item, gold_item, k, rank_keys, verbose=False):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, "k must be a positive integer grater than 0."

    rank = []
    num_distinct_evidence_sets = 0

    guess_ids = _get_ids_list(guess_item, rank_keys)[0]

    if guess_ids and len(guess_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in gold_item["output"]:
            if "provenance" in output:
                e_set = {
                    "+".join(
                        [
                            str(provenance[rank_key]).strip()
                            for rank_key in rank_keys
                            if rank_key in provenance
                        ]
                    )
                    for provenance in output["provenance"]
                }
                if e_set not in evidence_sets:  # no duplicate evidence set
                    evidence_sets.append(e_set)
                    e_size[len(e_set)] += 1
        num_distinct_evidence_sets = len(evidence_sets)

        # 2. check what's the minimum number of predicted pages needed to get a robust P/R@k
        min_prediction_size = 0
        c = 0
        for size, freq in sorted(e_size.items(), reverse=True):
            for _ in range(freq):
                min_prediction_size += size
                c += 1
                if c == k:
                    break
            if c == k:
                break
        # if the number of evidence sets is smaller than k
        min_prediction_size += k - c

        if verbose and len(guess_ids) < min_prediction_size:
            print(
                f"WARNING: you should provide at least {min_prediction_size} provenance items for a robust recall@{k} computation (you provided {len(guess_ids)} item(s))."
            )

        # 3. rank by gruping pages in each evidence set (each evidence set count as 1),
        # the position in the rank of each evidence set is given by the last page in guess_ids
        # non evidence pages counts as 1
        rank = []
        for guess_id in guess_ids:
            guess_id = str(guess_id).strip()
            found = False
            for idx, e_set in enumerate(evidence_sets):

                e_set_id = f"evidence_set:{idx}"

                if guess_id in e_set:
                    found = True

                    # remove from the rank previous points referring to this evidence set
                    if e_set_id in rank:
                        rank.remove(e_set_id)

                    # remove the guess_id from the evidence set
                    e_set.remove(guess_id)

                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
                    else:
                        # add a point for this partial evidence set
                        rank.append(e_set_id)

            if not found:
                rank.append(False)

    return rank, num_distinct_evidence_sets


# 1. Precision computation
def _precision_at_k(rank, k):

    # precision @ k
    p = rank[:k].count(True) / k

    return p


# 2. Recall computation
def _recall_at_k(rank, num_distinct_evidence_sets, k):

    r = rank[:k].count(True) / num_distinct_evidence_sets

    return r


# 3. Success rate computation
def _success_rate_at_k(rank, k):

    # success rate @ k
    p = int(True in rank[:k])

    return p


# 4. Answer in context computation
def _answer_in_context_at_k(guess_item, gold_item, k):

    answers = eval_downstream.get_gold_answers(gold_item)

    if "provenance" in guess_item["output"][0]:
        provenance = guess_item["output"][0]["provenance"]
        for i in range(0, min(k, len(provenance))):
            if "text" in provenance[i]:
                normalized_text = eval_downstream.normalize_answer(
                    provenance[i]["text"]
                )
                for a in answers:
                    if eval_downstream.normalize_answer(a) in normalized_text:
                        return 1
    return 0


def _computeRprec(guess_ids, gold_ids):

    R = len(gold_ids)
    num = 0

    for prediction in guess_ids[:R]:
        if str(prediction).strip() in gold_ids:
            num += 1

    Rprec = num / R if R > 0 else 0
    return Rprec


# R-precision https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_486
def rprecision(guess_item, gold_item, rank_keys):
    gold_ids_list = _get_ids_list(gold_item, rank_keys)
    guess_ids = _get_ids_list(guess_item, rank_keys)[0]
    Rprec_vector = []
    for gold_ids in gold_ids_list:
        Rprec = _computeRprec(guess_ids, gold_ids)
        Rprec_vector.append(Rprec)
    return max(Rprec_vector)


def get_ranking_metrics_for_item(guess_item, gold_item, ks, rank_keys, compute_rank_func):

    Rprec = 0
    P_at_k = {"precision@{}".format(k): 0 for k in sorted(ks) if k > 0}
    R_at_k = {"recall@{}".format(k): 0 for k in sorted(ks) if k > 1}
    S_at_k = {"success_rate@{}".format(k): 0 for k in sorted(ks) if k > 1}
    A_at_k = {"answer_in_context@{}".format(k): 0 for k in sorted(ks) if k > 0}

    assert (
        "output" in guess_item and len(guess_item["output"]) == 1
    ), f"guess should provide exactly one output for {guess_item['id']}"

    Rprec = rprecision(guess_item, gold_item, rank_keys=rank_keys)
    for k in ks:

        # 0. get rank
        rank, num_distinct_evidence_sets = compute_rank_func(
            guess_item, gold_item, k, rank_keys=rank_keys
        )

        if num_distinct_evidence_sets > 0:

            # 1. precision
            P_at_k["precision@{}".format(k)] = _precision_at_k(rank, k)

            # 2. recall
            R_at_k["recall@{}".format(k)] = _recall_at_k(
                rank, num_distinct_evidence_sets, k
            )

            # 3. success rate
            S_at_k["success_rate@{}".format(k)] = _success_rate_at_k(rank, k)

        # 4. answer in context
        A_at_k["answer_in_context@{}".format(k)] = _answer_in_context_at_k(
            guess_item, gold_item, k
        )

    return {"Rprec": Rprec, **P_at_k, **R_at_k, **S_at_k, **A_at_k}


def get_ranking_metrics(gold_dataset, guess_dataset, ks, rank_keys, compute_rank_func):

    ks = sorted([int(x) for x in ks])

    result = OrderedDict()
    result["Rprec"] = 0.0
    for k in ks:
        if k > 0:
            result["precision@{}".format(k)] = 0.0
            result["answer_in_context@{}".format(k)] = 0.0
        if k > 1:
            result["recall@{}".format(k)] = 0.0
            result["success_rate@{}".format(k)] = 0.0

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert (
            str(gold["id"]).strip() == str(guess["id"]).strip()
        ), "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics_for_item(guess_item, gold_item, ks, rank_keys, compute_rank_func)
        result["Rprec"] += ranking_metrics["Rprec"]
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] += ranking_metrics[
                    "precision@{}".format(k)
                ]
                result["answer_in_context@{}".format(k)] += ranking_metrics[
                    "answer_in_context@{}".format(k)
                ]
            if k > 1:
                result["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]
                result["success_rate@{}".format(k)] += ranking_metrics[
                    "success_rate@{}".format(k)
                ]

    if len(guess_dataset) > 0:
        result["Rprec"] /= len(guess_dataset)
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] /= len(guess_dataset)
                result["answer_in_context@{}".format(k)] /= len(guess_dataset)
            if k > 1:
                result["recall@{}".format(k)] /= len(guess_dataset)
                result["success_rate@{}".format(k)] /= len(guess_dataset)

    return result

def filter_answers(guess):
    new_guess = []
    for x in guess:
        new_o = []
        for o in x["output"]:
            if "provenance" in o:
                new_o.append(o)
        assert (
            len(new_o) == 1
        ), f"guess should provide exactly one output with provenance for {x['id']}"
        x["output"] = new_o
        new_guess.append(x)
    return new_guess


def evaluate(gold, guess, ks, rank_keys, compute_rank_for_item_func):
    pp = pprint.PrettyPrinter(indent=4)

    gold_dataset = kilt_utils.load_data(gold)
    guess_dataset = kilt_utils.load_data(guess)

    # 0. validate input
    gold_dataset, guess_dataset = eval_downstream.validate_input(
        gold_dataset, guess_dataset
    )

    # 1. filter out ground thruth answers
    guess_dataset = filter_answers(guess_dataset)

    # 2. get retrieval metrics
    result = get_ranking_metrics(gold_dataset, guess_dataset, ks, rank_keys, compute_rank_func=compute_rank_for_item_func)

    pp.pprint(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")
    parser.add_argument(
        "--ks",
        type=str,
        required=False,
        default="1,5,10,20",
        help="Comma separated list of positive integers for recall@k and precision@k",
    )
    parser.add_argument(
        "--rank_keys",
        type=str,
        required=False,
        default="wikipedia_id",
        help="Comma separated list of rank keys for recall@k and precision@k",
    )

    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]
    args.rank_keys = [rank_key for rank_key in args.rank_keys.split(",")]

    evaluate(args.gold, args.guess, args.ks, args.rank_keys, compute_rank_for_item_func=compute_rank_for_item)
