import unittest
import importlib.resources

import kilt.eval_downstream
import kilt.eval_retrieval
import tests.test_data as test_data


class TestEvalDownstream(unittest.TestCase):
    def test_calculate_metrics(self):

        with importlib.resources.open_text(test_data, "gold1.out") as gold_file:
            with importlib.resources.open_text(test_data, "guess1_1.out") as guess_file:
                result = kilt.eval_downstream.evaluate(gold_file.name, guess_file.name)
                self.assertEqual(result["downstream"]["em"], 2 / 3)
                self.assertEqual(result["downstream"]["f1"], 0.8333333333333334)
                self.assertEqual(result["downstream"]["rougel"], 0.8333333287500001)

            with importlib.resources.open_text(test_data, "gold1.out") as guess_file:
                result = kilt.eval_downstream.evaluate(gold_file.name, guess_file.name)
                self.assertEqual(result["downstream"]["em"], 1)
                self.assertEqual(result["downstream"]["f1"], 1)
                self.assertEqual(result["downstream"]["rougel"], 0.999999995)
