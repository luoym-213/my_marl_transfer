import unittest
from unittest import mock

import numpy as np
import torch

from learner import Learner
from multiagent.global_belief_map import GlobalBeliefMap
from planning.rrt_GNN import plan_batch
from utils import make_multiagent_env


class VectorizedRRTTest(unittest.TestCase):
    def test_batch_rrt_returns_fixed_finite_candidates(self):
        rng = np.random.RandomState(5)
        entropy_maps = rng.rand(4, 20, 20).astype(np.float32)
        voronoi_masks = rng.rand(4, 20, 20) > 0.5
        voronoi_masks[0] = False
        starts = np.array([[0, 0], [5, 7], [19, 19], [10, 10]])

        np.random.seed(11)
        result = plan_batch(
            starts,
            voronoi_masks,
            entropy_maps,
            max_iterations=12,
            top_k=5,
        )

        self.assertEqual(result.shape, (4, 5, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result[..., :2] >= 0) & (result[..., :2] < 20)).all())
        for row in result:
            self.assertEqual(len(set(map(tuple, row[:, :2]))), 5)

    def test_mixed_sampling_distribution_selects_expected_cells(self):
        entropy = np.array([[[0.0, np.log(3.0)]]], dtype=np.float32)
        entropy_maps = np.repeat(entropy, 2, axis=0)
        voronoi_masks = np.ones_like(entropy_maps, dtype=bool)
        starts = np.zeros((2, 2), dtype=np.int32)
        random_values = np.array([[0.2], [0.8]], dtype=np.float64)

        with mock.patch(
                'planning.rrt_GNN.np.random.random',
                return_value=random_values):
            result = plan_batch(
                starts,
                voronoi_masks,
                entropy_maps,
                radius=0,
                max_iterations=1,
                top_k=2,
                uniform_ratio=0.3,
            )

        # Mixture CDF for cell 0 is 0.3*0.5 + 0.7*0.25 = 0.325.
        # The first draw repeats the start; the second expands to cell 1.
        self.assertEqual(tuple(result[0, 0, :2]), (0.0, 0.0))
        self.assertEqual(tuple(result[1, 0, :2]), (0.0, 1.0))

    def test_zero_iterations_uses_fallback_and_exact_top_k(self):
        entropy_maps = np.zeros((1, 5, 5), dtype=np.float32)
        entropy_maps[0, 4, 4] = 1.0
        voronoi_masks = np.zeros_like(entropy_maps, dtype=bool)
        voronoi_masks[0, 4, 4] = True

        result = plan_batch(
            np.array([[2, 2]]),
            voronoi_masks,
            entropy_maps,
            max_iterations=0,
            top_k=5,
        )

        self.assertEqual(result.shape, (1, 5, 3))
        self.assertEqual(tuple(result[0, 0, :2]), (2.0, 2.0))
        self.assertEqual(tuple(result[0, 1, :2]), (4.0, 4.0))


class BatchedLandmarkTest(unittest.TestCase):
    def setUp(self):
        self.learner = Learner.__new__(Learner)
        self.learner.device = torch.device('cpu')

    def test_matching_cleanup_full_slots_and_team_broadcast(self):
        num_processes = 3
        num_agents = 2
        max_landmarks = 3
        data = torch.zeros(num_agents * num_processes, max_landmarks, 4)
        mask = torch.zeros(num_agents * num_processes, max_landmarks, 1)

        data[0, 0] = torch.tensor([0.0, 0.0, 2.0, 1.0])
        mask[0, 0] = 1.0
        data[1, 0] = torch.tensor([0.4, 0.4, 2.0, 1.0])
        mask[1, 0] = 1.0
        data[2, :, :2] = torch.tensor([
            [0.0, 0.0], [0.2, 0.0], [0.4, 0.0]
        ])
        data[2, :, 2] = 2.0
        mask[2, :, 0] = 1.0

        # Deliberately different teammate state must be overwritten by broadcast.
        data[num_processes:] = 99.0
        mask[num_processes:] = 1.0
        detected = [
            np.array([[0.05, 0.0], [0.8, 0.8]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([
                [0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.9, 0.9]
            ], dtype=np.float32),
        ]

        updated, updated_mask = self.learner.update_landmark_info(
            data, mask, detected, torch.device('cpu'),
            env_dones=torch.tensor([False, True, False]),
        )

        torch.testing.assert_close(updated[0, 0, :2], torch.tensor([0.035, 0.0]))
        torch.testing.assert_close(updated[0, 1], torch.tensor([0.8, 0.8, 2.0, 0.0]))
        self.assertEqual(updated[0, 0, 3].item(), 1.0)
        self.assertTrue(torch.equal(updated[1], torch.zeros_like(updated[1])))
        self.assertTrue(torch.equal(updated_mask[1], torch.zeros_like(updated_mask[1])))
        torch.testing.assert_close(updated[2, :, :2], data[2, :, :2])
        for agent_idx in range(1, num_agents):
            start = agent_idx * num_processes
            torch.testing.assert_close(updated[start:start + num_processes], updated[:num_processes])
            torch.testing.assert_close(updated_mask[start:start + num_processes], updated_mask[:num_processes])

    def test_later_detection_matches_landmark_inserted_earlier(self):
        data = torch.zeros(2, 3, 4)
        mask = torch.zeros(2, 3, 1)
        detected = [np.array([
            [0.2, 0.2], [0.21, 0.2]
        ], dtype=np.float32)]

        updated, updated_mask = self.learner.update_landmark_info(
            data, mask, detected, torch.device('cpu')
        )

        torch.testing.assert_close(
            updated[0, 0], torch.tensor([0.207, 0.2, 2.0, 0.0])
        )
        self.assertEqual(updated_mask[0, :, 0].tolist(), [1.0, 0.0, 0.0])
        torch.testing.assert_close(updated[0], updated[1])
        torch.testing.assert_close(updated_mask[0], updated_mask[1])


class MapTransportTest(unittest.TestCase):
    def test_landmark_heatmap_recomputed_when_positions_reset(self):
        belief_map = GlobalBeliefMap(
            landmark_positions=np.array([[-0.5, 0.0]], dtype=np.float32)
        )
        original_heatmap = belief_map.landmark_heatmap.copy()

        belief_map.reset(np.array([[0.5, 0.0]], dtype=np.float32))

        self.assertFalse(np.array_equal(original_heatmap, belief_map.landmark_heatmap))
        new_grid = belief_map.world_to_grid((0.5, 0.0))
        self.assertAlmostEqual(
            float(belief_map.landmark_heatmap[new_grid]),
            float(belief_map.landmark_heatmap.max()),
            places=6,
        )

    def test_learner_cache_only_refreshes_heatmaps_present_in_info(self):
        learner = Learner.__new__(Learner)
        learner.device = torch.device('cpu')
        learner.envs_info = None
        learner.landmark_heatmap_cache = None
        initial = [
            {'landmark_heatmap': np.full((2, 2), 1.0, dtype=np.float32)},
            {'landmark_heatmap': np.full((2, 2), 2.0, dtype=np.float32)},
        ]
        learner.set_envs_info(initial)
        data_ptr = learner.landmark_heatmap_cache.data_ptr()

        learner.set_envs_info([{'world_steps': 1}, {'world_steps': 1}])
        self.assertEqual(learner.landmark_heatmap_cache.data_ptr(), data_ptr)
        torch.testing.assert_close(
            learner.landmark_heatmap_cache[:, 0, 0], torch.tensor([1.0, 2.0])
        )

        learner.set_envs_info([
            {'world_steps': 2},
            {
                'world_steps': 0,
                'landmark_heatmap': np.full((2, 2), 7.0, dtype=np.float32),
            },
        ])
        torch.testing.assert_close(
            learner.landmark_heatmap_cache[:, 0, 0], torch.tensor([1.0, 7.0])
        )

    def test_environment_only_sends_landmark_heatmap_on_reset(self):
        env = make_multiagent_env(
            'simple_spread', 3, 0.1, 1, 0, mask_obs_dist=0.3
        )
        _, _, reset_info = env.reset()
        self.assertNotIn('belief_map', reset_info)
        self.assertIn('landmark_heatmap', reset_info)

        step_data = {
            'agents_actions': np.zeros(3, dtype=np.int64),
            'agents_goals': np.zeros((3, 2), dtype=np.float32),
            'agents_tasks': np.zeros((3, 1), dtype=np.int64),
        }
        _, _, _, _, step_info, _ = env.step(step_data)
        self.assertNotIn('belief_map', step_info)
        self.assertNotIn('landmark_heatmap', step_info)


if __name__ == '__main__':
    unittest.main()
