import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import estimator_policy
from estimator_policy import (
    estimator_eval_act,
    install_estimator_evaluator,
    rewrite_low_level_teammates,
)
from estimators.base import EstimateBatch, StateEstimator
from estimators.data import (
    dataset_metadata,
    hash_policy_state_dicts,
    save_trajectory_dataset,
    load_trajectory_dataset,
)
from estimators.training import calibrate_kf, train_commdrop


class FixedEstimator(StateEstimator):
    def __init__(self):
        super(FixedEstimator, self).__init__(2)

    def reset(self, initial_states, initial_semantics=None, source_step=0):
        self.was_reset = True

    def ingest_deliveries(self, deliveries):
        self.deliveries = deliveries

    def predict(self, query_step, ego_states, ego_semantics=None):
        mean = np.array([
            [[0.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.2, 0.3]],
            [[-0.7, -0.2, -0.1, 0.4], [0.1, 0.0, 0.0, 0.0]],
        ], dtype=np.float32)
        covariance = np.tile(np.eye(4), (2, 2, 1, 1)).astype(np.float32)
        return EstimateBatch(mean, covariance, np.zeros((2, 2)), np.zeros((2, 2)))


class FakeMap(object):
    def get_voronoi_region_masks(self, positions, dones):
        return np.ones((2, 2, 2), dtype=bool)


class IntegrationTest(unittest.TestCase):
    def test_low_level_rewrite_changes_only_teammate_segment(self):
        observations = np.arange(20, dtype=np.float32).reshape(2, 10)
        states = FixedEstimator().predict(0, np.zeros((2, 4))).mean
        rewritten = rewrite_low_level_teammates(observations, states)
        np.testing.assert_array_equal(rewritten[:, :8], observations[:, :8])
        np.testing.assert_allclose(rewritten[0, 8:10], [0.8, 0.1])
        np.testing.assert_allclose(rewritten[1, 8:10], [-0.8, -0.2])

    def test_estimate_enters_high_and_low_paths_without_truth(self):
        learner = type("LearnerDouble", (), {})()
        learner.env = type("EnvDouble", (), {
            "local_belief_maps": [FakeMap(), FakeMap()]
        })()
        learner.state_estimator = FixedEstimator()
        learner._estimator_query_step = None
        learner.envs_info = {
            "world_steps": 0,
            "local_agent_states": np.zeros((2, 2, 4), dtype=np.float32),
            "local_agent_tasks": np.zeros((2, 2, 1), dtype=np.int64),
            "local_agent_goals": np.zeros((2, 2, 2), dtype=np.float32),
            "local_task_progress": np.zeros((2, 2), dtype=np.float32),
            "local_agent_active": np.ones((2, 2), dtype=np.float32),
            "local_voronoi_masks": np.ones((2, 2, 2), dtype=bool),
            "delivery_events": [],
        }
        observations = np.zeros((2, 10), dtype=np.float32)
        captured = {}

        def fake_stale(instance, obs, *args, **kwargs):
            captured["states"] = instance.envs_info["local_agent_states"].copy()
            captured["obs"] = np.asarray(obs).copy()
            return "called"

        with mock.patch.object(estimator_policy, "stale_eval_act", fake_stale):
            result = estimator_eval_act(
                learner, observations, np.zeros(1), torch.ones(2, 1),
                torch.zeros(2, 2), torch.zeros(2, 1),
                torch.zeros(2, 2, 4), torch.zeros(2, 2, 1),
            )
        self.assertEqual(result, "called")
        np.testing.assert_allclose(captured["states"][0, 1, :2], [0.8, 0.1])
        np.testing.assert_allclose(captured["obs"][0, 8:10], [0.8, 0.1])
        self.assertNotIn("ground_truth_agent_states", learner.envs_info)

    def test_policy_hash_is_unchanged_by_installation(self):
        state = [{"weight": torch.arange(4, dtype=torch.float32)}]
        before = hash_policy_state_dicts(state)
        learner = type("LearnerDouble", (), {})()
        install_estimator_evaluator(learner, FixedEstimator())
        after = hash_policy_state_dicts(state)
        self.assertEqual(before, after)

    def test_small_data_calibrate_train_evaluate_smoke(self):
        episodes = []
        for index in range(6):
            time_count = 4
            states = np.zeros((time_count, 3, 4), dtype=np.float32)
            states[:, :, 0] = np.arange(time_count)[:, None] * 0.02
            states[:, :, 1] = np.arange(3)[None, :] * 0.05
            states[:, :, 2] = 0.2
            episodes.append({
                "episode_index": index,
                "trajectory_seed": 7378 + index,
                "communication_seed": 20260905 + index,
                "dt": 0.1,
                "states": states,
                "goals": np.zeros((time_count, 3, 2), dtype=np.float32),
                "options": np.zeros((time_count, 3), dtype=np.int64),
                "task_progress": np.zeros((time_count, 3), dtype=np.float32),
                "active": np.ones((time_count, 3), dtype=np.float32),
                "landmarks": np.zeros((3, 2), dtype=np.float32),
            })
        dataset = dataset_metadata(6, "/tmp/frozen.pt", "hash", dt=0.1)
        dataset["episodes"] = episodes
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "data.pt")
            save_trajectory_dataset(dataset, data_path)
            loaded = load_trajectory_dataset(data_path)
            kf_path = os.path.join(directory, "kf.pt")
            calibrate_kf(loaded, "mild", kf_path, multipliers=(1.0,))
            self.assertTrue(os.path.isfile(kf_path))
            commdrop_path = os.path.join(directory, "commdrop.pt")
            checkpoint = train_commdrop(
                loaded, "mild", commdrop_path,
                network_config={"hidden_dim": 8, "task_dim": 4},
                episode_batch_size=2, max_epochs=1, patience=1,
            )
            self.assertTrue(os.path.isfile(commdrop_path))
            self.assertEqual(checkpoint["best_validation_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
