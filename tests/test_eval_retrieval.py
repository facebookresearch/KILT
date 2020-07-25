import unittest
import importlib.resources

import kilt.eval_downstream
import kilt.eval_retrieval
import tests.test_data as test_data


class TestEvalRetrieval(unittest.TestCase):
    def test_calculate_metrics(self):

        with importlib.resources.open_text(test_data, "gold2.out") as gold_file:
            with importlib.resources.open_text(test_data, "guess2_1.out") as guess_file:
                result = kilt.eval_retrieval.evaluate(
                    gold_file.name,
                    guess_file.name,
                    ks=[1, 5],
                    rank_keys=["wikipedia_id"],
                )
                self.assertEqual(result["Rprec"], 1 / 2)
                self.assertEqual(result["precision@1"], 1)
                self.assertEqual(result["precision@5"], 2 / 5)
                self.assertEqual(result["recall@5"], 2 / 3)
                self.assertEqual(result["success_rate@5"], 1)

                result = kilt.eval_retrieval.evaluate(
                    gold_file.name,
                    guess_file.name,
                    ks=[1, 5],
                    rank_keys=["wikipedia_id", "section"],
                )
                self.assertEqual(result["Rprec"], 1 / 2)
                self.assertEqual(result["precision@1"], 1)
                self.assertEqual(result["precision@5"], 2 / 5)
                self.assertEqual(result["recall@5"], 2 / 3)
                self.assertEqual(result["success_rate@5"], 1)

                result = kilt.eval_retrieval.evaluate(
                    gold_file.name,
                    guess_file.name,
                    ks=[1, 5],
                    rank_keys=[
                        "wikipedia_id",
                        "start_paragraph_id",
                        "end_paragraph_id",
                    ],
                )
                self.assertEqual(result["Rprec"], 1 / 2)
                self.assertEqual(result["precision@1"], 1)
                self.assertEqual(result["precision@5"], 2 / 5)
                self.assertEqual(result["recall@5"], 2 / 3)
                self.assertEqual(result["success_rate@5"], 1)
