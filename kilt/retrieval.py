# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import os.path
from os import path

from kilt import kilt_utils as utils


def generate_output_file(output_folder, dataset_file):
    basename = os.path.basename(dataset_file)
    output_file = os.path.join(output_folder, basename)
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

    for task_family, datasets in test_config_json.items():
        logger.info("TASK: {}".format(task_family))

        for dataset_name, dataset_file in datasets.items():
            logger.info("DATASET: {}".format(dataset_name))

            if dataset_file:

                output_file = generate_output_file(output_folder, dataset_file)
                if path.exists(output_file):
                    logger.info(
                        "Skip output file {} that already exists.".format(output_file)
                    )
                    continue

                raw_data = utils.load_data(dataset_file)

                # consider only valid data - filter out invalid
                validated_data = {}
                query_data = []
                for element in raw_data:
                    #if utils.validate_datapoint(element, logger=None):
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
                ranker.feed_data(query_data)
                provenance = ranker.run()

                if len(provenance) != len(query_data):
                    logger.warning(
                        "different numbers of queries: {} and predicions: {}".format(
                            len(query_data), len(provenance)
                        )
                    )

                # write prediction files
                if provenance:
                    logger.info("writing prediction file to {}".format(output_file))

                    predictions = []
                    for query_id in provenance.keys():
                        element = validated_data[query_id]
                        new_output = [{"provenance": provenance[query_id]}]
                        # append the answers
                        if "output" in element:
                            for o in element["output"]:
                                if "answer" in o:
                                    new_output.append({"answer": o["answer"]})
                        element["output"] = new_output
                        predictions.append(element)

                    with open(output_file, "w+") as outfile:
                        for p in predictions:
                            json.dump(p, outfile)
                            outfile.write("\n")
