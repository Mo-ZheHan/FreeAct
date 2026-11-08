"""Check the target suites: tasks, instructions, episode horizon and initial states."""

import unittest

import freeact.envs  # noqa: F401  (registers the target suites)
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite

SUITES = {"libero_target": (5, "milk carton"), "libero_camera": (1, "milk carton"), "libero_bottle": (1, "ketchup bottle")}


class SuiteTest(unittest.TestCase):
    def test_target_suites(self):
        for name, (n_tasks, obj) in SUITES.items():
            with self.subTest(suite=name):
                suite = _get_suite(name)
                self.assertEqual(suite.n_tasks, n_tasks)
                self.assertEqual(TASK_SUITE_MAX_STEPS[name], 500)
                for i in range(n_tasks):
                    self.assertIn(obj, suite.get_task(i).language)
                    self.assertEqual(len(suite.get_task_init_states(i)), 50)


if __name__ == "__main__":
    unittest.main()
