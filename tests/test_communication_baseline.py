import unittest

import numpy as np

from communication_baseline import (
    CommunicationDegradedEnv,
    make_communication_env,
    make_ideal_env,
)
from multiagent.communication import (
    COMMUNICATION_PROFILES,
    CommunicationSimulator,
)
from multiagent.global_belief_map import GlobalBeliefMap


class CommunicationProfileTest(unittest.TestCase):
    def test_profiles_match_chapter_three_table(self):
        self.assertEqual(
            COMMUNICATION_PROFILES["mild"],
            {
                "radius": 1.5,
                "mean_delay": 1,
                "max_delay": 2,
                "loss_probability": 0.05,
            },
        )
        self.assertEqual(
            COMMUNICATION_PROFILES["medium"],
            {
                "radius": 0.9,
                "mean_delay": 2,
                "max_delay": 5,
                "loss_probability": 0.15,
            },
        )
        self.assertEqual(
            COMMUNICATION_PROFILES["severe"],
            {
                "radius": 0.6,
                "mean_delay": 4,
                "max_delay": 10,
                "loss_probability": 0.30,
            },
        )

    def test_packets_are_directed_and_delayed(self):
        channel = CommunicationSimulator(
            2,
            seed=3,
            profile={
                "radius": 1.0,
                "mean_delay": 1,
                "max_delay": 2,
                "loss_probability": 0.0,
            },
        )
        channel.schedule_broadcasts(
            4,
            np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
            [{"sender": 0}, {"sender": 1}],
        )
        self.assertEqual(channel.deliver(4), [])
        delivered = channel.deliver(5)
        self.assertEqual(len(delivered), 2)
        self.assertEqual(
            {(event["sender"], event["receiver"]) for event in delivered},
            {(0, 1), (1, 0)},
        )

    def test_out_of_range_packets_are_not_queued(self):
        channel = CommunicationSimulator(
            2,
            seed=3,
            profile={
                "radius": 0.5,
                "mean_delay": 1,
                "max_delay": 1,
                "loss_probability": 0.0,
            },
        )
        channel.schedule_broadcasts(
            1,
            np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            [{}, {}],
        )
        self.assertEqual(channel.deliver(10), [])
        self.assertEqual(channel.get_stats()["out_of_range"], 2)

    def test_packet_loss_prevents_delivery(self):
        channel = CommunicationSimulator(
            2,
            seed=3,
            profile={
                "radius": 10.0,
                "mean_delay": 1,
                "max_delay": 1,
                "loss_probability": 1.0,
            },
        )
        channel.schedule_broadcasts(
            1,
            np.zeros((2, 2), dtype=np.float32),
            [{}, {}],
        )
        self.assertEqual(channel.deliver(10), [])
        self.assertEqual(channel.get_stats()["lost"], 2)


class LocalInformationTest(unittest.TestCase):
    def test_belief_fusion_is_idempotent(self):
        local_map = GlobalBeliefMap(
            world_size=2.0, cell_size=0.1, landmark_positions=[]
        )
        received = np.full_like(local_map.belief_grid, 0.5)
        received[2, 3] = 0.8

        CommunicationDegradedEnv._merge_belief_grid(local_map, received)
        first = local_map.belief_grid.copy()
        CommunicationDegradedEnv._merge_belief_grid(local_map, received)

        self.assertAlmostEqual(float(local_map.belief_grid[2, 3]), 0.8)
        np.testing.assert_array_equal(local_map.belief_grid, first)

    def test_environment_keeps_stale_state_and_independent_maps(self):
        env = make_communication_env(
            "simple_spread",
            num_agents=3,
            dist_threshold=0.1,
            arena_size=1,
            identity_size=0,
            mask_obs_dist=0.5,
            communication_level="severe",
            communication_seed=17,
        )
        env.seed(11)
        _, _, reset_info = env.reset()
        initially_known_agent_one = reset_info[
            "local_agent_states"
        ][0, 1].copy()

        positions = np.array([
            [-0.9, -0.9],
            [0.9, 0.9],
            [0.9, -0.9],
        ], dtype=np.float32)
        for agent, position in zip(env.agents, positions):
            agent.state.p_pos = position.copy()
            agent.state.p_vel = np.zeros(2, dtype=np.float32)

        step_data = {
            "agents_actions": np.zeros(3, dtype=np.int64),
            "agents_goals": positions.copy(),
            "agents_tasks": np.zeros((3, 1), dtype=np.int64),
        }
        observations, _, _, _, info, _ = env.step(step_data)

        np.testing.assert_allclose(
            info["local_agent_states"][0, 1], initially_known_agent_one
        )
        self.assertFalse(np.allclose(
            info["local_agent_states"][0, 1, :2],
            env.agents[1].state.p_pos,
        ))

        agent_one_grid = env.local_belief_maps[1].world_to_grid(
            env.agents[1].state.p_pos
        )
        self.assertNotAlmostEqual(
            float(env.local_belief_maps[1].belief_grid[agent_one_grid]), 0.5
        )
        self.assertAlmostEqual(
            float(env.local_belief_maps[0].belief_grid[agent_one_grid]), 0.5
        )

        # The low-level observation sees the same stale teammate state rather
        # than reconstructing truth from another agent's live observation.
        other_start = 4 + 2 * env.n
        expected_relative = (
            initially_known_agent_one[:2] - env.agents[0].state.p_pos
        )
        np.testing.assert_allclose(
            observations[0][other_start:other_start + 2],
            expected_relative,
            atol=1e-6,
        )

    def test_reset_does_not_inherit_previous_episode_goals_or_global_maps(self):
        env = make_communication_env(
            "simple_spread",
            num_agents=3,
            dist_threshold=0.1,
            arena_size=1,
            identity_size=0,
            communication_level="mild",
            communication_seed=17,
        )
        env.seed(11)
        env.reset()
        for agent in env.agents:
            agent.state.g_pos = np.array([7.0, -7.0], dtype=np.float32)

        _, _, info = env.reset()
        positions = np.asarray(
            [agent.state.p_pos for agent in env.agents], dtype=np.float32
        )
        expected_goals = np.repeat(positions[None, :, :], env.n, axis=0)
        np.testing.assert_allclose(info["local_agent_goals"], expected_goals)
        for agent in env.agents:
            np.testing.assert_allclose(agent.state.g_pos, agent.state.p_pos)

        for key in ("map", "entropy_map", "voronoi_masks", "heatmap"):
            self.assertNotIn(key, info)
        self.assertIn("landmark_heatmap", info)
        self.assertFalse(np.any(info["landmark_heatmap"]))

    def test_delayed_packet_keeps_its_source_timestamp_state(self):
        env = make_communication_env(
            "simple_spread",
            num_agents=2,
            dist_threshold=0.1,
            arena_size=1,
            identity_size=0,
            communication_level="mild",
            communication_seed=17,
        )
        env.seed(11)
        env.reset()
        env.communication.profile.update({
            "radius": 10.0,
            "mean_delay": 1,
            "max_delay": 1,
            "loss_probability": 0.0,
        })
        env.communication._sample_delay = lambda: 1

        first_positions = np.array(
            [[-0.4, -0.4], [0.4, 0.4]], dtype=np.float32
        )
        for agent, position in zip(env.agents, first_positions):
            agent.state.p_pos = position.copy()
            agent.state.p_vel = np.zeros(2, dtype=np.float32)
        step_data = {
            "agents_actions": np.zeros(2, dtype=np.int64),
            "agents_goals": first_positions.copy(),
            "agents_tasks": np.zeros((2, 1), dtype=np.int64),
        }
        env.step(step_data)
        source_state = env.local_agent_states[1, 1].copy()

        env.agents[1].state.p_pos = np.array([0.8, -0.8], dtype=np.float32)
        second_positions = np.asarray(
            [agent.state.p_pos for agent in env.agents], dtype=np.float32
        )
        step_data["agents_goals"] = second_positions.copy()
        _, _, _, _, info, _ = env.step(step_data)

        self.assertEqual(int(info["local_source_steps"][0, 1]), 1)
        np.testing.assert_allclose(info["local_agent_states"][0, 1], source_state)
        self.assertFalse(np.allclose(
            info["local_agent_states"][0, 1, :2],
            env.agents[1].state.p_pos,
        ))
        for key in (
                "map", "entropy_map", "voronoi_masks", "heatmap",
                "landmark_heatmap"):
            self.assertNotIn(key, info)

    def test_episode_seeding_matches_initial_scenes_across_conditions(self):
        common = {
            "env_id": "simple_spread",
            "num_agents": 3,
            "dist_threshold": 0.1,
            "arena_size": 1,
            "identity_size": 0,
            "mask_obs_dist": 0.5,
            "trajectory_seed": 7378,
        }
        ideal = make_ideal_env(**common)
        degraded = make_communication_env(
            communication_level="severe",
            communication_seed=20260905,
            **common
        )
        ideal.seed(7378)
        degraded.seed(7378)

        for _ in range(3):
            ideal.reset()
            degraded.reset()
            ideal_entities = np.asarray([
                entity.state.p_pos for entity in ideal.world.entities
            ])
            degraded_entities = np.asarray([
                entity.state.p_pos for entity in degraded.world.entities
            ])
            np.testing.assert_allclose(ideal_entities, degraded_entities)


if __name__ == "__main__":
    unittest.main()
