from __future__ import annotations

import unittest

from cgr_exp.datasets.game24 import Game24Action, Game24Task
from cgr_exp.verification.game24_verifier import Game24Verifier


class Game24StateTest(unittest.TestCase):
    def test_exact_transition_consumes_two_numbers_and_adds_result(self) -> None:
        task = Game24Task(task_id="g24-demo", numbers=(4, 5, 6, 10))
        root = task.root_node()
        action = Game24Action(left=10, right=4, operator="-")
        result = Game24Verifier().verify(task, root, action)
        child = Game24Verifier().make_child(task, root, action, result, child_index=0)
        self.assertTrue(result.feasible)
        self.assertEqual(child.structured_state["values"], [5.0, 6.0, 6.0])
        self.assertEqual(child.structured_state["expression_count"], 3)

    def test_checker_rejects_reusing_unavailable_number_and_bad_arithmetic(self) -> None:
        task = Game24Task(task_id="g24-demo", numbers=(4, 5, 6, 10))
        root = task.root_node()
        verifier = Game24Verifier()
        reused = verifier.verify(task, root, Game24Action(left=10, right=10, operator="+"))
        unknown = verifier.verify(task, root, Game24Action(left=7, right=4, operator="+"))
        divide_zero = verifier.verify(task, root, Game24Action(left=4, right=0, operator="/"))
        self.assertFalse(reused.feasible)
        self.assertIn("number-not-available", reused.violations)
        self.assertFalse(unknown.feasible)
        self.assertIn("number-not-available", unknown.violations)
        self.assertFalse(divide_zero.feasible)
        self.assertIn("divide-by-zero", divide_zero.violations)


if __name__ == "__main__":
    unittest.main()
