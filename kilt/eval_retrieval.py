import argparse
from collections import defaultdict

from kilt import kilt_utils
from kilt import eval_downstream


def _get_gold_ids_list(datapoint):
    # collect all gold ids
    gold_wikipedia_ids_list = []
    for output in datapoint["output"]:
        single_wikipedia_ids_list = []
        for provenance in output["provenance"]:
            single_wikipedia_ids_list.append(str(provenance["wikipedia_id"]))
        gold_wikipedia_ids_list.append(list(set(single_wikipedia_ids_list)))

    # consider only unique ids

    return gold_wikipedia_ids_list


def get_rank(datapoint, predicted_page_ids, k):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, "k must be a positive integer grater than 0."

    rank = []
    num_distinct_evidence_sets = 0

    if predicted_page_ids and len(predicted_page_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in datapoint["output"]:
            e_set = {
                str(provenance["wikipedia_id"]).strip()
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
        assert (
            len(predicted_page_ids) >= min_prediction_size
        ), f"you should provide at least {min_prediction_size} predicted pages for a robust recall@{k} computation"

        # 3. rank by gruping pages in each evidence set (each evidence set count as 1),
        # the position in the rank of each evidence set is given by the last page in predicted_page_ids
        # non evidence pages counts as 1
        rank = []
        for page in predicted_page_ids:
            page = str(page).strip()
            found = False
            for idx, e_set in enumerate(evidence_sets):

                e_set_id = f"evidence_set:{idx}"

                if page in e_set:
                    found = True

                    # remove from the rank previous points referring to this evidence set
                    if e_set_id in rank:
                        rank.remove(e_set_id)

                    # remove the page from the evidence set
                    e_set.remove(page)

                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
                    else:
                        # add a point for this partial evidence set
                        rank.append(e_set_id)

            if not found:
                rank.append(False)

    return rank, num_distinct_evidence_sets


def precision_at_k(datapoint, predicted_page_ids, k):

    rank, _ = get_rank(datapoint, predicted_page_ids, k)

    # precision @ k
    p = rank[:k].count(True) / k

    return p


def recall_at_k(datapoint, predicted_page_ids, k):

    rank, num_distinct_evidence_sets = get_rank(datapoint, predicted_page_ids, k)

    r = 0

    if num_distinct_evidence_sets > 0:
        # recall @ k
        r = rank[:k].count(True) / num_distinct_evidence_sets
    else:
        print("WARNING: the number of distinct evidence sets is 0 for", datapoint)

    return r


def _computeMAP(gold_wikipedia_ids, predicted_page_ids, topk):
    # Mean Average Precision
    rate_num = 0.0
    rate_den = 0.0
    MAP = 0.0
    for prediction in predicted_page_ids:
        rate_den += 1
        if prediction in gold_wikipedia_ids:
            rate_num += 1
            # fraction of relevant documents so far
            MAP += rate_num / rate_den

    # less than topk relevant documents could be available
    MAP /= min(topk, len(gold_wikipedia_ids))

    return MAP


def meanAvgPrecision(datapoint, predicted_page_ids, topk=100):
    # return the maximum value among all the possible answer's provenance list
    gold_wikipedia_ids_list = _get_gold_ids_list(datapoint)
    MAP_vector = []
    for gold_wikipedia_ids in gold_wikipedia_ids_list:
        MAP = _computeMAP(gold_wikipedia_ids, predicted_page_ids, topk)
        MAP_vector.append(MAP)
    return max(MAP_vector)


def _computeRprec(gold_wikipedia_ids, predicted_page_ids):
    R = len(gold_wikipedia_ids)
    num = 0

    deduplicated_page_ids = []

    for x in predicted_page_ids:
        if x not in deduplicated_page_ids:
            deduplicated_page_ids.append(x)
            # consider only the first R predictions
            if len(deduplicated_page_ids) == R:
                break

    assert len(deduplicated_page_ids) <= R

    for prediction in deduplicated_page_ids:
        if str(prediction).strip() in gold_wikipedia_ids:
            num += 1

    Rprec = num / R
    return Rprec


# R-precision https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_486
def rprecision(datapoint, predicted_page_ids):

    gold_wikipedia_ids_list = _get_gold_ids_list(datapoint)
    Rprec_vector = []
    for gold_wikipedia_ids in gold_wikipedia_ids_list:
        Rprec = _computeRprec(gold_wikipedia_ids, predicted_page_ids)
        Rprec_vector.append(Rprec)
    return max(Rprec_vector)


def get_ranking_metrics(guess_item, gold_item, ks):

    Rprec = 0
    R_at_k = {"recall@{}".format(k): 0 for k in ks}
    P_at_k = {"precision@{}".format(k): 0 for k in ks}

    assert (
        "output" in guess_item and len(guess_item["output"]) == 1
    ), "guess should provide exactly one output"

    output = guess_item["output"][0]
    guess_wikipedia_ids = []
    if "provenance" in output:
        for provenance in output["provenance"]:
            if "wikipedia_id" not in provenance:
                print(
                    "wikipedia_id not in provenance for-> {}\n skipping".format(
                        provenance
                    )
                )
            else:
                guess_wikipedia_ids.append(provenance["wikipedia_id"])

    if len(guess_wikipedia_ids) > 0:
        Rprec = rprecision(gold_item, guess_wikipedia_ids)
        for k in ks:
            R_at_k["recall@{}".format(k)] = recall_at_k(
                gold_item, guess_wikipedia_ids, k
            )
            P_at_k["precision@{}".format(k)] = precision_at_k(
                gold_item, guess_wikipedia_ids, k
            )

    return {"Rprec": Rprec, **R_at_k, **P_at_k}


def compute(gold_dataset, guess_dataset, ks):

    Rprec = 0.0
    R_at_k = {"recall@{}".format(k): 0 for k in ks}
    P_at_k = {"precision@{}".format(k): 0 for k in ks}

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert (
            str(gold["id"]).strip() == str(guess["id"]).strip()
        ), "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks)
        Rprec += ranking_metrics["Rprec"]
        for k in ks:
            R_at_k["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]
            P_at_k["precision@{}".format(k)] += ranking_metrics[
                "precision@{}".format(k)
            ]

    if len(guess_dataset) > 0:
        Rprec /= len(guess_dataset)
        for k in ks:
            R_at_k["recall@{}".format(k)] /= len(guess_dataset)
            P_at_k["precision@{}".format(k)] /= len(guess_dataset)

    return {"Rprec": Rprec, **R_at_k, **P_at_k}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10,20",
        help="Comma separated list of positive integers for recall@k and precision@k",
    )

    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]

    gold_dataset = kilt_utils.load_data(args.gold)
    guess_dataset = kilt_utils.load_data(args.guess)

    # 0. validate input
    # gold_dataset, guess_dataset = eval_downstream.validate_input(
    #    gold_dataset, guess_dataset
    # )

    print(compute(gold_dataset, guess_dataset, args.ks))
