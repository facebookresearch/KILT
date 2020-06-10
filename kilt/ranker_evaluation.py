import json
import sys

import pprint
import prettytable
from tqdm import tqdm
import os

from kilt import kilt_utils as utils
from kilt import eval_retrieval


def generate_output_file(output_folder, output_name, model_name, dataset_file):
    basename = os.path.basename(dataset_file)
    output_file = os.path.join(output_folder, output_name, model_name, basename)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    return output_file


def run(
    test_config_json,
    ranker,
    model_name,
    logger,
    topk=100,
    debug=False,
    output_folder="",
):

    if debug:
        pp = pprint.PrettyPrinter(indent=4)

    table = prettytable.PrettyTable(
        ["TASK", "DATASET", "P@1", "R-prec", "MAP", "support"]
    )
    result = {}

    for task_family, datasets in test_config_json.items():
        logger.info("TASK: {}".format(task_family))
        for dataset_name, dataset_file in datasets.items():
            logger.info("DATASET: {}".format(dataset_name))
            if dataset_file:
                raw_data = utils.load_data(dataset_file)

                # consider only valid data - filter out invalid
                validated_data = {}
                query_data = []
                for element in raw_data:
                    if utils.validate_datapoint(element, logger=None):
                        if element["id"] in validated_data:
                            raise ValueError("ids are not unique in input data!")
                        validated_data[element["id"]] = element
                        query_data.append(
                            {"query": element["input"], "id": element["id"]}
                        )

                if debug:
                    # just consider the top10 datapoints
                    query_data = query_data[:10]
                    print("query_data: {}", format(query_data))

                # get predictions
                ranker.fed_data(query_data, topk)
                all_doc_id, all_doc_scores, all_query_id, meta = ranker.run()

                if len(all_query_id) != len(query_data):
                    logger.warning(
                        "different numbers of queries: {} and predicions: {}".format(
                            len(query_data), len(all_query_id)
                        )
                    )

                # write retrieved augmented data - for dpr, blink, drqa
                if meta:
                    output_file = generate_output_file(
                        output_folder,
                        "retrieved_augmented_datasets",
                        model_name,
                        dataset_file,
                    )

                    print(
                        "writing retrieved augmented output in {}".format(output_file),
                        flush=True,
                    )
                    rad = []
                    for query_id in all_query_id:
                        element = validated_data[query_id]
                        element["retrieved"] = meta[query_id]
                        rad.append(element)
                    with open(output_file, "w+") as outfile:
                        for rdata in rad:
                            json.dump(rdata, outfile)
                            outfile.write("\n")

                # write predictions files for BLINK
                if False and model_name == "blink" and meta:
                    output_file = generate_output_file(
                        output_folder, "predictions", model_name, dataset_file
                    )

                    print(
                        "writing predictions in {}".format(output_file), flush=True,
                    )
                    predictions = []
                    for query_id in all_query_id:
                        element = validated_data[query_id]
                        retrieved = meta[query_id]
                        predictions.append(
                            {
                                "id": element["id"],
                                "input": element["input"],
                                "output": [
                                    {
                                        "answer": retrieved[0]["wikipedia_title"],
                                        "provenance": [retrieved],
                                    }
                                ],
                            }
                        )

                    with open(output_file, "w+") as outfile:
                        for prediction in predictions:
                            json.dump(prediction, outfile)
                            outfile.write("\n")

                # evaluate
                global_p1 = 0.0
                global_Rprec = 0.0
                global_MAP = 0.0

                for doc_names, doc_scores, query_id in zip(
                    all_doc_id, all_doc_scores, all_query_id
                ):
                    element = validated_data[query_id]

                    if doc_names and len(doc_names) > 0:
                        local_p1 = eval_retrieval.precision_at_1(element, doc_names)
                        local_Rprec = eval_retrieval.rprecision(element, doc_names)

                        global_p1 += local_p1
                        global_Rprec += local_Rprec

                        if doc_names and len(doc_names) >= topk:

                            local_MAP = eval_retrieval.meanAvgPrecision(
                                element, doc_names, topk
                            )

                            global_MAP += local_MAP

                            if debug:
                                pp.pprint(element)
                                print("doc_names : {}".format(doc_names))
                                print("p1 : {}".format(local_p1))
                                print("Rprec : {}".format(local_Rprec))
                                print("MAP : {}".format(local_MAP))
                                input("...")
                        else:
                            pass
                            """
                            logger.warning(
                                "less than top-{} results returned: {} for query: {}".format(
                                    topk, len(doc_names), element["input"]
                                )
                            )
                            """
                    else:
                        logger.error(
                            "no results returned: {} for query: {}".format(
                                len(doc_names), element["input"]
                            )
                        )

                global_p1 /= len(query_data)
                global_Rprec /= len(query_data)
                global_MAP /= len(query_data)

                logger.info("p@1: {}%".format(round(global_p1 * 100, 2)))
                logger.info("Rprec: {}%".format(round(global_Rprec * 100, 2)))
                logger.info("MAP: {}%".format(round(global_MAP * 100, 2)))
                logger.info("support: {}".format(len(all_query_id)))

                table.add_row(
                    [
                        task_family,
                        dataset_name,
                        round(global_p1 * 100, 2),
                        round(global_Rprec * 100, 2),
                        round(global_MAP * 100, 2),
                        len(all_query_id),
                    ]
                )

                result[dataset_name] = {
                    "p@1": global_p1,
                    "Rprec": global_Rprec,
                    "MAP": global_MAP,
                }

            else:
                logger.warning("skip - missing test file")

    logger.info("\n{}".format(table))

    return result
