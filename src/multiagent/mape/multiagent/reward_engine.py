import numpy as np


class RewardEngine:
    """Computes low-level shaping and high-level aggregate rewards."""

    def __init__(
        self,
        time_penalty=0.2,
        safe_distance=0.15,
        collision_coef=-20.0,
        boundary_penalty=-2.0,
    ):
        self.time_penalty = time_penalty
        self.safe_distance = safe_distance
        self.collision_coef = collision_coef
        self.boundary_penalty = boundary_penalty

    def collision_boundary_penalties(self, agent_positions, world_size):
        num_agents = len(agent_positions)
        penalties = np.zeros(num_agents)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                if dist < self.safe_distance:
                    collision_penalty = self.collision_coef * (
                        (1 - dist / self.safe_distance) ** 2
                    )
                    penalties[i] += collision_penalty
                    penalties[j] += collision_penalty

        boundary = world_size / 2.0
        for i, (x, y) in enumerate(agent_positions):
            if abs(x) >= boundary or abs(y) >= boundary:
                penalties[i] += self.boundary_penalty

        return penalties

    def low_level_rewards(self, current_goal_rewards, last_goal_rewards, penalties):
        return np.array(current_goal_rewards) - np.array(last_goal_rewards) + penalties

    def high_level_rewards(
        self,
        explore_rewards,
        discover_target_rewards,
        reach_target_rewards,
    ):
        return (
            np.array(explore_rewards)
            + np.array(discover_target_rewards)
            + np.array(reach_target_rewards)
            - self.time_penalty
        )
