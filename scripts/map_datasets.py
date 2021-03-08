# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from kilt import dataset_mapper
from kilt.datasets import (
    base_dataset,
    entity_linking,
    fact_verification,
    natural_questions,
    zero_shot_re,
    hotpotqa,
    wizard,
)


if __name__ == "__main__":
    datasets = []

    # NQ dev set
    datasets.append(
        natural_questions.NaturalQuestionsDataset.from_config_file(
            "dev_natural_questions", "kilt/configs/mapping/dev_natural_questions.json"
        )
    )

    for dataset in datasets:
        dataset_mapper.map_dataset(dataset=dataset)
