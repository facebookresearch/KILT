# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import importlib.resources

import kilt.eval_downstream
import kilt.eval_retrieval
import tests.test_data as test_data


class TestEvalDownstream(unittest.TestCase):
    def test_calculate_metrics(self):

        with importlib.resources.open_text(test_data, "gold1.jsonl") as gold_file:
            with importlib.resources.open_text(
                test_data, "guess1_1.jsonl"
            ) as guess_file:
                result = kilt.eval_downstream.evaluate(gold_file.name, guess_file.name)

                # kilt
                self.assertEqual(result["kilt"]["KILT-em"], 1 / 3)
                self.assertEqual(result["kilt"]["KILT-f1"], 1 / 3)
                self.assertEqual(result["kilt"]["KILT-rougel"], 0.3333333316666667)

                # downsream
                self.assertEqual(result["downstream"]["em"], 2 / 3)
                self.assertEqual(result["downstream"]["f1"], 0.8333333333333334)
                self.assertEqual(result["downstream"]["rougel"], 0.7222222178240741)

                # retrieval page level
                self.assertEqual(result["retrieval"]["Rprec"], 1 / 3)
                self.assertEqual(result["retrieval"]["recall@5"], 1 / 3)

            with importlib.resources.open_text(test_data, "gold1.jsonl") as guess_file:
                result = kilt.eval_downstream.evaluate(gold_file.name, guess_file.name)

                # kilt
                self.assertEqual(result["kilt"]["KILT-em"], 1)
                self.assertEqual(result["kilt"]["KILT-f1"], 1)
                self.assertEqual(result["kilt"]["KILT-rougel"], 0.999999995)

                # downsream
                self.assertEqual(result["downstream"]["em"], 1)
                self.assertEqual(result["downstream"]["f1"], 1)
                self.assertEqual(result["downstream"]["rougel"], 0.999999995)

                # retrieval page level
                self.assertEqual(result["retrieval"]["Rprec"], 1)
                self.assertEqual(result["retrieval"]["recall@5"], 1)

        with importlib.resources.open_text(test_data, "gold3.jsonl") as gold_file:
            with importlib.resources.open_text(
                test_data, "guess3_1.jsonl"
            ) as guess_file:

                result = kilt.eval_downstream.evaluate(gold_file.name, guess_file.name)

                # kilt
                self.assertEqual(result["kilt"]["KILT-em"], 0)
                self.assertEqual(result["kilt"]["KILT-f1"], 0.25510204081632654)
                self.assertEqual(result["kilt"]["KILT-rougel"], 0.22352940932318338)

                # downsream
                self.assertEqual(result["downstream"]["em"], 0)
                self.assertEqual(result["downstream"]["f1"], 0.5102040816326531)
                self.assertEqual(result["downstream"]["rougel"], 0.44705881864636676)
