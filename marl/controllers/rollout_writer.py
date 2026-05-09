import torch


class RolloutWriter:
    """Writes batched policy outputs back to agent rollout-facing fields."""

    def __init__(self, device):
        self.device = device

    def assign_act_outputs(
        self,
        team,
        low_outputs,
        goals,
        tasks,
        goal_log_probs,
        high_values,
        critic_map,
        critic_nodes,
        ego_nodes,
        explore_nodes,
        teammate_nodes,
        teammate_masks,
        landmark_data,
        landmark_mask,
        landmark_nodes,
    ):
        num_agents = len(team)
        values, actions, action_log_probs = [
            torch.chunk(output, num_agents) for output in low_outputs
        ]

        goals_split = torch.chunk(goals, num_agents)
        tasks_split = torch.chunk(tasks, num_agents)
        goal_log_probs_split = torch.chunk(goal_log_probs, num_agents)
        high_values_split = torch.chunk(high_values, num_agents, dim=1)
        ego_nodes_split = torch.chunk(ego_nodes, num_agents)
        explore_nodes_split = torch.chunk(explore_nodes, num_agents)
        teammate_nodes_split = torch.chunk(teammate_nodes, num_agents)
        teammate_masks_split = torch.chunk(teammate_masks, num_agents)
        landmark_data_split = torch.chunk(landmark_data, num_agents)
        landmark_mask_split = torch.chunk(landmark_mask, num_agents)
        landmark_nodes_split = torch.chunk(landmark_nodes, num_agents)

        actions_list = []
        goals_list = []
        tasks_list = []

        for i, agent in enumerate(team):
            agent.value = values[i]
            agent.action = actions[i]
            agent.action_log_prob = action_log_probs[i]

            agent.critic_map = critic_map
            agent.critic_nodes = critic_nodes
            agent.goal = goals_split[i]
            agent.task = tasks_split[i]
            agent.higoal_log_prob = goal_log_probs_split[i]
            agent.high_value = high_values_split[i]

            agent.ego_nodes = ego_nodes_split[i]
            agent.explore_nodes = explore_nodes_split[i]
            agent.landmark_data = landmark_data_split[i]
            agent.landmark_mask = landmark_mask_split[i]
            agent.teammate_nodes = teammate_nodes_split[i]
            agent.teammate_masks = teammate_masks_split[i]
            agent.landmark_nodes = landmark_nodes_split[i]

            actions_list.append(actions[i].cpu().numpy())
            goals_list.append(goals_split[i].cpu().numpy())
            tasks_list.append(tasks_split[i].cpu().numpy())

        return actions_list, goals_list, tasks_list

    def insert_transition(
        self,
        agents,
        obs,
        reward,
        high_rewards,
        masks,
        env_state,
        goal_dones,
    ):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        env_state_t = torch.from_numpy(env_state).float().to(self.device)
        for i, agent in enumerate(agents):
            agent_obs = obs_t[:, i, :]
            agent.update_rollout(
                agent_obs,
                reward[:, i].unsqueeze(1),
                high_rewards[:, i].unsqueeze(1),
                masks[:, i].unsqueeze(1),
                env_state_t,
                goal_dones[:, i].unsqueeze(1),
            )
