import unittest

import numpy as np

from estimator_communication import make_estimator_communication_env
from estimators.data import make_episode_splits, simulate_delivery_events


def sample_episode(index=0, time_count=8):
    states = np.zeros((time_count, 3, 4), dtype=np.float32)
    for step in range(time_count):
        states[step, :, 0] = np.array([0.0, 0.1, 0.2]) + step * 0.01
    return {
        "episode_index": index,
        "dt": 0.1,
        "states": states,
        "goals": np.zeros((time_count, 3, 2), dtype=np.float32),
        "options": np.zeros((time_count, 3), dtype=np.int64),
        "task_progress": np.zeros((time_count, 3), dtype=np.float32),
        "active": np.ones((time_count, 3), dtype=np.float32),
        "landmarks": np.zeros((3, 2), dtype=np.float32),
    }


class EstimatorDataTest(unittest.TestCase):
    def test_formal_split_is_strict_and_disjoint(self):
        split = make_episode_splits(1000)
        self.assertEqual([len(split[key]) for key in (
            "train", "validation", "test")], [700, 150, 150])
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_channel_replay_is_causal_and_reproducible(self):
        episode = sample_episode()
        first = simulate_delivery_events(episode, "mild", communication_seed=55)
        second = simulate_delivery_events(episode, "mild", communication_seed=55)
        signature = lambda events: [[(
            event["receiver"], event["sender"], event["source_step"],
            event["arrival_step"], tuple(event["motion_state"]),
        ) for event in step] for step in events]
        self.assertEqual(signature(first), signature(second))
        for query_step, events in enumerate(first):
            for event in events:
                self.assertEqual(event["arrival_step"], query_step)
                self.assertGreater(event["arrival_step"], event["source_step"])
                np.testing.assert_array_equal(
                    event["ego_source_motion_state"],
                    episode["states"][event["source_step"], event["receiver"]],
                )

    def test_environment_exposes_only_real_arrivals_by_ego(self):
        env = make_estimator_communication_env(
            "simple_spread", 2, 0.1, 1, 0,
            communication_level="mild", communication_seed=3,
        )
        env.seed(8)
        _, _, reset_info = env.reset()
        self.assertEqual(reset_info["delivery_events"], [])
        env.communication.profile.update({
            "radius": 10.0, "mean_delay": 1,
            "max_delay": 1, "loss_probability": 0.0,
        })
        data = {
            "agents_actions": np.zeros(2, dtype=np.int64),
            "agents_goals": np.asarray([
                agent.state.p_pos for agent in env.agents
            ], dtype=np.float32),
            "agents_tasks": np.zeros((2, 1), dtype=np.int64),
        }
        env.step(data)
        _, _, _, _, info, _ = env.step(data)
        self.assertEqual(len(info["delivery_events"]), 2)
        self.assertEqual([len(items) for items in info[
            "delivery_events_by_ego"]], [1, 1])
        required = {
            "sender", "receiver", "source_step", "arrival_step",
            "source_time", "arrival_time", "motion_state", "option",
            "goal", "task_progress", "active", "ego_source_motion_state",
        }
        for event in info["delivery_events"]:
            self.assertTrue(required.issubset(event))
            self.assertNotEqual(event["sender"], event["receiver"])


if __name__ == "__main__":
    unittest.main()
