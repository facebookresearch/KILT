import argparse

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


def recall_at_k(datapoint, predicted_page_ids, k=1):
    assert k > 1, "k must be a positive integer grater than 1."

    r = 0
    if predicted_page_ids and len(predicted_page_ids) > 0:
        top_k = {str(e) for e in predicted_page_ids[:k]}

        recalls = []
        for output in datapoint["output"]:
            relevant_set = {
                str(provenance["wikipedia_id"]).strip()
                for provenance in output["provenance"]
            }
            recalls.append(len(relevant_set.intersection(top_k)) / len(relevant_set))

        r = max(recalls)

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
    recall_at_k = {"recall@{}".format(k): 0 for k in ks}

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
            recall_at_k["recall@{}".format(k)] = recall_at_k(
                datapoint, predicted_page_ids, k
            )

    return {"p1": p1, "Rprec": Rprec, **recall_at_k}


def compute(gold_dataset, guess_dataset, ks):

    p1 = 0.0
    Rprec = 0.0
    recall_at_k = {"recall@{}".format(k): 0 for k in ks}

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert gold["id"] == guess["id"], "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks)
        p1 += ranking_metrics["p1"]
        Rprec += ranking_metrics["Rprec"]
        for k in recall_at_k:
            recall_at_k["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]

    p1 /= len(guess_dataset)
    Rprec /= len(guess_dataset)
    for k in recall_at_k:
        recall_at_k["recall@{}".format(k)] /= len(guess_dataset)

    return {"p1": p1, "Rprec": Rprec, **recall_at_k}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10,20",
        help="Comma separated list of positive integers for recall@k",
    )

    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]

    gold_dataset = kilt_utils.load_data(args.gold)
    guess_dataset = kilt_utils.load_data(args.guess)

    print(compute(gold_dataset, guess_dataset, args.ks))
