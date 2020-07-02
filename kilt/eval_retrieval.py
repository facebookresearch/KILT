import argparse
from collections import defaultdict

from kilt import kilt_utils


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


def precision_at_1(datapoint, predicted_page_ids):
    p = 0
    if predicted_page_ids and len(predicted_page_ids) > 0:
        winner_id = predicted_page_ids[0]
        for output in datapoint["output"]:
            for provenance in output["provenance"]:
                if str(provenance["wikipedia_id"]).strip() == str(winner_id).strip():
                    p = 1
                    return p
    return p


def recall_at_k(datapoint, predicted_page_ids, k):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, "k must be a positive integer grater than 0."

    r = 0
    if predicted_page_ids and len(predicted_page_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in datapoint["output"]:
            e_set = {
                str(provenance["wikipedia_id"]).strip()
                for provenance in output["provenance"]
            }
            if e_set in evidence_sets:
                print(
                    "WARNING, evidence set {} is already in evidence_sets {}".format(
                        e_set, evidence_sets
                    )
                )
                pass
            else:
                evidence_sets.append(e_set)
                e_size[len(e_set)] += 1

        denominator = len(evidence_sets)

        # 2. check what's the minimum number of predicted pages needed to get a robust R@k
        min_prediction_size = 0
        c = 0
        for size, freq in sorted(e_size.items(), reverse=True):
            print(size, freq)
            for _ in range(freq):
                min_prediction_size += size
                c += 1
                if c == k:
                    break
            if c == k:
                break

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
            for e_set in evidence_sets:
                if page in e_set:
                    found = True
                    e_set.remove(page)
                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
            if not found:
                rank.append(False)

        # 4. recall @ k
        r = sum(rank[:k]) / denominator

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

    p1 = 0
    Rprec = 0
    R_at_k = {"recall@{}".format(k): 0 for k in ks}

    assert (
        "output" in guess_item and len(guess_item["output"]) == 1
    ), "guess should provide exactly one output"
    output = guess_item["output"][0]
    guess_wikipedia_ids = []
    for provenance in output["provenance"]:
        if "wikipedia_id" not in provenance:
            print(
                "wikipedia_id not in provenance for-> {}\n skipping".format(provenance)
            )
        else:
            guess_wikipedia_ids.append(provenance["wikipedia_id"])

    if len(guess_wikipedia_ids) > 0:
        p1 = precision_at_1(gold_item, guess_wikipedia_ids)
        Rprec = rprecision(gold_item, guess_wikipedia_ids)
        for k in ks:
            R_at_k["recall@{}".format(k)] = recall_at_k(
                gold_item, guess_wikipedia_ids, k
            )

    return {"p1": p1, "Rprec": Rprec, **R_at_k}


def compute(gold_dataset, guess_dataset, ks):

    p1 = 0.0
    Rprec = 0.0
    R_at_k = {"recall@{}".format(k): 0 for k in ks}

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert gold["id"] == guess["id"], "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks)
        p1 += ranking_metrics["p1"]
        Rprec += ranking_metrics["Rprec"]
        for k in ks:
            R_at_k["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]

    p1 /= len(guess_dataset)
    Rprec /= len(guess_dataset)
    for k in ks:
        R_at_k["recall@{}".format(k)] /= len(guess_dataset)

    return {"p1": p1, "Rprec": Rprec, **R_at_k}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma separated list of positive integers for recall@k",
    )

    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]

    gold_dataset = kilt_utils.load_data(args.gold)
    guess_dataset = kilt_utils.load_data(args.guess)

    print(compute(gold_dataset, guess_dataset, args.ks))
