import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gym.spaces import Discrete

from mpnn import MPNN
from multiagent.global_belief_map import GlobalBeliefMap
from rlcore.algo.ppo import (
    JointPPO,
    build_smdp_decision_dataset,
)


def make_graph_batch(batch_size=3, num_agents=3, top_k=5, max_landmarks=3):
    ego_nodes = torch.randn(batch_size, 5)
    teammate_nodes = torch.randn(batch_size, num_agents, 5)
    teammate_masks = torch.ones(batch_size, num_agents, 1)
    explore_nodes = torch.randn(batch_size, top_k, 4)
    landmark_nodes = torch.randn(batch_size, max_landmarks, 4)
    landmark_nodes[:, :, 3] = 0.0
    landmark_masks = torch.zeros(batch_size, max_landmarks, 1)
    landmark_masks[:, :2] = 1.0
    return (
        ego_nodes,
        teammate_nodes,
        teammate_masks,
        explore_nodes,
        landmark_nodes,
        landmark_masks,
    )


def make_rollout(decision_indices, num_steps=4, num_processes=2,
                 num_agents=3, top_k=2):
    max_landmarks = num_agents
    goal_dones = torch.zeros(num_steps + 1, num_processes, 1)
    flat_goal_dones = goal_dones[:-1].reshape(-1)
    for index in decision_indices:
        flat_goal_dones[index] = 1.0

    return SimpleNamespace(
        high_rewards=torch.zeros(num_steps, num_processes, 1),
        critic_maps=torch.randn(num_steps, num_processes, 3, 8, 8),
        critic_nodes=torch.randn(
            num_steps, num_processes, num_agents, 4
        ),
        goals=torch.randn(num_steps, num_processes, 2),
        higoal_log_probs=torch.randn(num_steps, num_processes, 1),
        high_values=torch.randn(num_steps + 1, num_processes, 1),
        high_returns=torch.randn(num_steps + 1, num_processes, 1),
        goal_dones=goal_dones,
        ego_nodes=torch.randn(num_steps, num_processes, 5),
        explore_nodes=torch.randn(
            num_steps, num_processes, top_k, 4
        ),
        landmark_masks=torch.zeros(
            num_steps + 1, num_processes, max_landmarks, 1
        ),
        landmark_nodes=torch.randn(
            num_steps, num_processes, max_landmarks, 4
        ),
        teammate_nodes=torch.randn(
            num_steps, num_processes, num_agents, 5
        ),
        teammate_masks=torch.ones(
            num_steps, num_processes, num_agents, 1
        ),
    )


class HighLevelPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.num_agents = 3
        self.model = MPNN(
            Discrete(5),
            num_agents=self.num_agents,
            num_entities=self.num_agents,
            input_size=16,
            is_recurrent=False,
        )

    def _distribution_inputs(self):
        graph = make_graph_batch(num_agents=self.num_agents)
        (
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            landmark_nodes,
            landmark_masks,
        ) = graph
        (
            explore_edges,
            landmark_edges,
            landmark_edge_masks,
        ) = self.model.get_edge_features(
            explore_nodes, landmark_nodes, landmark_masks
        )
        return graph + (
            explore_edges,
            landmark_edges,
            landmark_edge_masks,
        )

    def test_ego_query_key_receive_actor_gradients(self):
        (
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            landmark_nodes,
            landmark_masks,
            explore_edges,
            landmark_edges,
            landmark_edge_masks,
        ) = self._distribution_inputs()

        result = self.model.get_high_level_goal(
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            explore_edges,
            landmark_nodes,
            landmark_masks,
            landmark_edges,
            landmark_edge_masks,
            deterministic=True,
        )
        (-result['node_log_probs'].mean()).backward()

        for module_name in (
                'ego_node_encoder',
                'q_proj',
                'k_proj',
                'v_proj',
                'candidate_fusion',
                'node_selection_head'):
            gradients = [
                parameter.grad
                for parameter in self.model.modules_dict['high_level'][
                    module_name
                ].parameters()
                if parameter.grad is not None
            ]
            gradient_norm = sum(
                gradient.abs().sum().item() for gradient in gradients
            )
            self.assertGreater(
                gradient_norm, 0.0,
                '{} did not receive actor gradient'.format(module_name)
            )

    def test_sampling_and_ppo_recompute_have_same_log_probability(self):
        (
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            landmark_nodes,
            landmark_masks,
            explore_edges,
            landmark_edges,
            landmark_edge_masks,
        ) = self._distribution_inputs()

        with torch.no_grad():
            result = self.model.get_high_level_goal(
                ego_nodes,
                teammate_nodes,
                teammate_masks,
                explore_nodes,
                explore_edges,
                landmark_nodes,
                landmark_masks,
                landmark_edges,
                landmark_edge_masks,
                deterministic=True,
            )
            critic_maps = torch.randn(ego_nodes.size(0), 3, 100, 100)
            critic_nodes = torch.randn(
                ego_nodes.size(0), self.num_agents, 4
            )
            agent_ids = torch.tensor([0, 1, 2])
            (
                selected_values,
                recomputed_log_probs,
                _
            ) = self.model.evaluate_high_actions(
                critic_maps,
                critic_nodes,
                result['waypoints'],
                ego_nodes,
                explore_nodes,
                landmark_masks,
                landmark_nodes,
                teammate_nodes,
                teammate_masks,
                agent_ids,
            )

            all_values = self.model.get_high_value(
                critic_maps, critic_nodes
            )
            expected_values = all_values[
                torch.arange(agent_ids.numel()), agent_ids
            ].unsqueeze(-1)

        self.assertTrue(torch.allclose(
            recomputed_log_probs,
            result['node_log_probs'],
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            selected_values,
            expected_values,
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_invalid_candidates_are_masked(self):
        (
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            landmark_nodes,
            landmark_masks,
            explore_edges,
            landmark_edges,
            _,
        ) = self._distribution_inputs()

        node_dist, _, unified_mask, _, _ = (
            self.model._build_high_node_distribution(
                ego_nodes,
                teammate_nodes,
                teammate_masks,
                explore_nodes,
                explore_edges,
                landmark_nodes,
                landmark_masks,
                landmark_edges,
            )
        )
        self.assertTrue(torch.all(node_dist.probs[~unified_mask] == 0))

    def test_all_targeted_landmarks_fall_back_to_explore(self):
        graph = make_graph_batch(num_agents=self.num_agents)
        (
            ego_nodes,
            teammate_nodes,
            teammate_masks,
            explore_nodes,
            landmark_nodes,
            landmark_masks,
        ) = graph
        landmark_masks.fill_(1.0)
        landmark_nodes[:, :, 3] = 1.0
        explore_edges, landmark_edges, _ = self.model.get_edge_features(
            explore_nodes, landmark_nodes, landmark_masks
        )

        node_dist, _, unified_mask, _, _ = (
            self.model._build_high_node_distribution(
                ego_nodes,
                teammate_nodes,
                teammate_masks,
                explore_nodes,
                explore_edges,
                landmark_nodes,
                landmark_masks,
                landmark_edges,
            )
        )

        top_k = explore_nodes.size(1)
        self.assertTrue(unified_mask[:, :top_k].all())
        self.assertFalse(unified_mask[:, top_k:].any())
        self.assertTrue(torch.isfinite(node_dist.probs).all())
        self.assertTrue(torch.allclose(
            node_dist.probs.sum(dim=1),
            torch.ones(ego_nodes.size(0)),
        ))


class SMDPDecisionDatasetTest(unittest.TestCase):
    def test_only_decision_points_are_materialized(self):
        rollouts = [
            make_rollout([0, 5]),
            make_rollout([3]),
        ]
        advantages = [
            rollout.high_returns[:-1] - rollout.high_values[:-1]
            for rollout in rollouts
        ]
        decision_data = build_smdp_decision_dataset(
            rollouts, advantages
        )

        self.assertEqual(decision_data[0].size(0), 3)
        self.assertEqual(decision_data[7].tolist(), [0, 0, 1])
        self.assertTrue(torch.isfinite(decision_data[6]).all())
        self.assertAlmostEqual(
            decision_data[6].mean().item(), 0.0, places=6
        )

    def test_single_decision_advantage_is_finite(self):
        rollout = make_rollout([0])
        advantages = [
            rollout.high_returns[:-1] - rollout.high_values[:-1]
        ]
        decision_data = build_smdp_decision_dataset(
            [rollout], advantages
        )
        self.assertTrue(torch.isfinite(decision_data[6]).all())
        self.assertEqual(decision_data[6].item(), 0.0)

    def test_no_decisions_skip_joint_ppo_update(self):
        torch.manual_seed(11)
        model = MPNN(
            Discrete(5),
            num_agents=3,
            num_entities=3,
            input_size=16,
            is_recurrent=False,
        )
        trainer = JointPPO(
            model,
            clip_param=0.2,
            ppo_epoch=4,
            num_mini_batch=4,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            lr=3e-4,
            max_grad_norm=0.5,
        )
        rollout = make_rollout([])
        self.assertEqual(
            trainer.update([rollout]),
            (0.0, 0.0, 0.0),
        )


    def test_joint_ppo_updates_a_decision_batch(self):
        torch.manual_seed(13)
        model = MPNN(
            Discrete(5),
            num_agents=3,
            num_entities=3,
            input_size=16,
            is_recurrent=False,
        )
        trainer = JointPPO(
            model,
            clip_param=0.2,
            ppo_epoch=1,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            lr=3e-4,
            max_grad_norm=0.5,
        )
        rollout = make_rollout(
            [0], num_steps=1, num_processes=1
        )
        rollout.critic_maps = torch.randn(1, 1, 3, 100, 100)
        rollout.landmark_nodes[..., 3] = 0.0

        losses = trainer.update([rollout])

        self.assertEqual(len(losses), 3)
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        low_level_ids = {
            id(parameter)
            for parameter in model.get_module_params('low_level')
        }
        self.assertTrue(all(
            id(parameter) not in low_level_ids
            for parameter in trainer.trainable_params
        ))


class RewardConsolidationTest(unittest.TestCase):
    def test_combined_reward_matches_compatibility_interfaces(self):
        kwargs = dict(
            world_size=2.0,
            cell_size=0.02,
            landmark_positions=np.array([
                [0.1, 0.1],
                [-0.4, 0.3],
            ]),
            landmark_radius=0.05,
            obs_radius=0.3,
        )
        agent_positions = np.array([
            [0.0, 0.0],
            [0.4, -0.2],
        ])
        compatibility_map = GlobalBeliefMap(**kwargs)
        combined_map = GlobalBeliefMap(**kwargs)

        explore_rewards = (
            compatibility_map.get_agent_step_explore_entropy(
                agent_positions, 0.3
            )
        )
        discover_rewards = (
            compatibility_map.get_agent_discover_target_reward(
                agent_positions, 0.3
            )
        )
        combined_rewards = combined_map.get_agent_step_rewards(
            agent_positions, 0.3
        )

        np.testing.assert_allclose(
            combined_rewards['explore_rewards'],
            explore_rewards,
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            combined_rewards['discover_rewards'],
            discover_rewards,
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            compatibility_map.belief_grid,
            combined_map.belief_grid,
        )


if __name__ == '__main__':
    unittest.main()
