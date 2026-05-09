import numpy as np


def get_agent_step_rewards(map_state, agent_positions, radius, discover_reward_scale=1.0):
    if len(agent_positions) == 0:
        return {"explore_rewards": [], "discover_rewards": []}

    original_belief_grid = map_state.belief_grid.copy()
    original_entropy_map = map_state.compute_shannon_entropy()
    original_total_entropy = np.sum(original_entropy_map)

    explore_rewards = []
    discover_rewards = []

    for agent_pos in agent_positions:
        map_state.belief_grid = original_belief_grid.copy()

        fov_mask = map_state.get_fov_mask(agent_pos, radius)
        positive_mask = fov_mask & map_state.landmark_map
        negative_mask = fov_mask & (~map_state.landmark_map)

        map_state.bayesian_update(positive_mask, negative_mask)

        updated_entropy_map = map_state.compute_shannon_entropy()
        updated_total_entropy = np.sum(updated_entropy_map)
        explore_reward = original_total_entropy - updated_total_entropy
        explore_rewards.append(
            float(explore_reward / map_state.explore_reward_normalization) * 2
        )

        delta_belief = map_state.belief_grid - original_belief_grid
        positive_delta = np.maximum(0, delta_belief[fov_mask])
        total_discover = np.sum(positive_delta)
        discover_reward = total_discover * discover_reward_scale
        discover_rewards.append(
            float(discover_reward / map_state.discover_reward_normalization) * 5
        )

    map_state.belief_grid = original_belief_grid

    return {
        "explore_rewards": explore_rewards,
        "discover_rewards": discover_rewards,
    }


def get_agent_step_explore_entropy(
    map_state,
    agent_positions,
    radius,
    sigma=None,
    clip_outside=True,
):
    return get_agent_step_rewards(map_state, agent_positions, radius)["explore_rewards"]


def get_agent_discover_target_reward(
    map_state,
    agent_positions,
    radius,
    reward_value=1.0,
):
    return get_agent_step_rewards(
        map_state, agent_positions, radius, reward_value
    )["discover_rewards"]
