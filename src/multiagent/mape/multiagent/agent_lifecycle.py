import numpy as np


class AgentLifecycle:
    """Tracks per-episode collection state for high-level goals."""

    def __init__(self, num_agents, landmark_positions=None):
        self.num_agents = num_agents
        self.reset(landmark_positions)

    def reset(self, landmark_positions=None):
        self.landmark_positions = landmark_positions if landmark_positions is not None else []
        self.visited_landmarks = set()
        self.agents_done = [False] * self.num_agents

    def goal_dones(self, agents, dist_threshold):
        goal_dones = [
            np.linalg.norm(agent.state.p_pos - agent.state.g_pos) < dist_threshold
            for agent in agents
        ]
        for i in range(len(goal_dones)):
            if self.agents_done[i]:
                goal_dones[i] = False
        return goal_dones

    def collect_rewards(
        self,
        agents,
        agents_task,
        goal_dones,
        dist_threshold,
        target_reward=10.0,
    ):
        rewards = []

        for agent_idx, agent in enumerate(agents):
            agent_reward = 0.0

            if self.agents_done[agent_idx]:
                rewards.append(agent_reward)
                continue

            agent_task = agents_task[agent_idx]
            task_value = agent_task[0] if isinstance(agent_task, (list, np.ndarray)) else agent_task
            if task_value != 1:
                rewards.append(agent_reward)
                continue

            if not goal_dones[agent_idx]:
                rewards.append(agent_reward)
                continue

            current_goal = agent.state.g_pos
            min_dist = float("inf")
            matched_landmark_idx = None

            for landmark_idx, landmark_pos in enumerate(self.landmark_positions):
                dist = np.linalg.norm(current_goal - landmark_pos)
                if dist < min_dist:
                    min_dist = dist
                    matched_landmark_idx = landmark_idx

            if min_dist < dist_threshold and matched_landmark_idx not in self.visited_landmarks:
                reward_multiplier = sum(self.agents_done) + 1
                agent_reward = target_reward * reward_multiplier
                self.visited_landmarks.add(matched_landmark_idx)
                self.agents_done[agent_idx] = True

            rewards.append(agent_reward)

        return rewards
