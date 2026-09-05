import os
import unittest
from unittest import mock

import numpy as np
from gym.spaces import Discrete

from gym_vecenv.subproc_vec_env import (
    SubprocVecEnv,
    _restore_cpu_affinity,
)


class _MinimalEnv:
    observation_space = Discrete(2)
    action_space = Discrete(2)
    num_agents = 1

    def reset(self):
        return np.zeros(1), np.zeros(1), {}


def _make_minimal_env():
    return _MinimalEnv()


@unittest.skipUnless(
    hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'),
    'Linux CPU affinity APIs are required',
)
class SubprocAffinityTest(unittest.TestCase):
    def test_workers_restore_parent_affinity(self):
        expected_affinity = set(os.sched_getaffinity(0))
        envs = SubprocVecEnv([_make_minimal_env, _make_minimal_env])
        try:
            envs.reset()
            for process in envs.ps:
                self.assertEqual(
                    set(os.sched_getaffinity(process.pid)),
                    expected_affinity,
                )
        finally:
            envs.close()

    def test_workers_respect_parent_affinity_subset(self):
        original_affinity = set(os.sched_getaffinity(0))
        if len(original_affinity) < 2:
            self.skipTest('At least two CPUs are required')

        restricted_affinity = set(sorted(original_affinity)[-2:])
        envs = None
        try:
            os.sched_setaffinity(0, restricted_affinity)
            envs = SubprocVecEnv([_make_minimal_env, _make_minimal_env])
            envs.reset()
            for process in envs.ps:
                self.assertEqual(
                    set(os.sched_getaffinity(process.pid)),
                    restricted_affinity,
                )
        finally:
            if envs is not None:
                envs.close()
            os.sched_setaffinity(0, original_affinity)

    def test_affinity_failure_warns_and_falls_back(self):
        with mock.patch.object(
                os, 'sched_setaffinity', side_effect=OSError('test failure')):
            with self.assertWarns(RuntimeWarning):
                restored = _restore_cpu_affinity(os.sched_getaffinity(0))

        self.assertFalse(restored)


if __name__ == '__main__':
    unittest.main()
