import numpy as np
import torch

from marl.controllers.high_level_context import (
    agent_batteries,
    build_agent_context,
    detected_maps as build_detected_maps,
    goal_done_mask as build_goal_done_mask,
    stack_global_maps,
    voronoi_masks as build_voronoi_masks,
)


class EvalController:
    """Evaluation-time action selection for a single environment."""

    def __init__(self, high_level_policy, device):
        self.high_level_policy = high_level_policy
        self.device = device

    def act(
        self,
        learner,
        obs,
        env_states,
        masks,
        goals,
        tasks,
        landmark_data,
        landmark_mask,
        deterministic=True,
    ):
        grouped_obs = self._group_observations_by_team(learner, obs)
        actions = []
        env_states = torch.from_numpy(env_states).float().to(self.device)

        for team, policy, team_obs in zip(
            learner.teams_list, learner.policies_list, grouped_obs
        ):
            all_goals = goals
            all_tasks = tasks
            num_agents = len(team)
            obs_tensor = torch.cat(team_obs, dim=0).to(self.device)

            envs_info = [learner.envs_info]
            entropy_maps, _, _, _ = stack_global_maps(envs_info, self.device)
            current_goal_done_mask = build_goal_done_mask(envs_info, self.device)
            batteries = agent_batteries(envs_info, num_agents, self.device)

            detected_maps = build_detected_maps(envs_info, self.device)
            new_detected, new_detected_masks = learner.update_landmark_info(
                landmark_data,
                landmark_mask,
                detected_maps,
                self.device,
            )

            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(
                1, num_agents, 1, 1
            )
            voronoi_masks = build_voronoi_masks(envs_info, num_agents, self.device)
            agent_context = build_agent_context(
                obs_tensor,
                all_goals,
                masks,
                batteries,
                num_agents,
                1,
            )

            decision = self.high_level_policy.select_goals(
                policy=policy,
                obs_positions=obs_tensor[:, 2:4],
                all_masks=None,
                goals=all_goals,
                tasks=all_tasks,
                goal_log_probs=None,
                goal_done_mask=current_goal_done_mask,
                agent_entropy_maps=agent_entropy_maps,
                voronoi_masks=voronoi_masks,
                agent_nodes=agent_context["agent_nodes"],
                ego_nodes=agent_context["ego_nodes"],
                teammate_nodes=agent_context["teammate_nodes"],
                teammate_mask=agent_context["teammate_mask"],
                landmark_data=new_detected,
                landmark_mask=new_detected_masks,
                num_processes=1,
                deterministic=deterministic,
                update_tasks=True,
                update_log_probs=False,
                update_targeted=True,
                store_landmark_nodes=False,
            )
            landmark_data = decision["landmark_data"]
            landmark_mask = decision["landmark_mask"]

            if len(team_obs) != 0:
                _, action, _ = policy.low_level_act(
                    obs_tensor, all_goals, deterministic=True
                )
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions), all_goals, all_tasks, landmark_data, landmark_mask

    def _group_observations_by_team(self, learner, obs):
        obs1 = []
        obs2 = []
        grouped_obs = []
        for i in range(len(obs)):
            agent = learner.env.world.policy_agents[i]
            obs_tensor = torch.as_tensor(
                obs[i], dtype=torch.float, device=self.device
            ).view(1, -1)
            if hasattr(agent, "adversary") and agent.adversary:
                obs1.append(obs_tensor)
            else:
                obs2.append(obs_tensor)
        if len(obs1) != 0:
            grouped_obs.append(obs1)
        if len(obs2) != 0:
            grouped_obs.append(obs2)
        return grouped_obs
