import numpy as np
import torch

from marl.controllers.high_level_context import (
    goal_done_mask as build_goal_done_mask,
)
from marl.controllers.local_maps import flatten_agent_major


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
        landmark_timestamp,
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

            envs_info = [learner.envs_info] if isinstance(learner.envs_info, dict) else learner.envs_info
            learner.envs_info = envs_info
            if envs_info[0].get("world_steps", 0) == 0:
                learner.local_map_bank.reset()

            current_goal_done_mask = build_goal_done_mask(envs_info, self.device)
            env_dones = learner._env_dones_from_masks(masks, num_agents, 1)
            prepared = learner._prepare_local_policy_inputs(
                obs_tensor,
                all_goals,
                masks,
                landmark_data,
                landmark_mask,
                landmark_timestamp,
                num_agents,
                1,
                env_dones,
            )

            new_detected = prepared["landmark_data"]
            new_detected_masks = prepared["landmark_mask"]
            new_detected_timestamps = prepared["landmark_timestamp"]
            agent_context = prepared["agent_context"]

            decision = self.high_level_policy.select_goals(
                policy=policy,
                obs_positions=obs_tensor[:, 2:4],
                all_masks=None,
                goals=all_goals,
                tasks=all_tasks,
                goal_log_probs=None,
                goal_done_mask=current_goal_done_mask,
                agent_entropy_maps=prepared["agent_entropy_maps"],
                voronoi_masks=prepared["local_voronoi_masks"],
                agent_nodes=agent_context["agent_nodes"],
                ego_nodes=agent_context["ego_nodes"],
                teammate_nodes=agent_context["teammate_nodes"],
                teammate_mask=prepared["high_teammate_masks"],
                landmark_data=new_detected,
                landmark_mask=new_detected_masks,
                goal_visibility_mask=prepared["comm_mask"],
                num_processes=1,
                deterministic=deterministic,
                update_tasks=True,
                update_log_probs=False,
                update_targeted=True,
                store_landmark_nodes=False,
            )
            landmark_data = decision["landmark_data"]
            landmark_mask = decision["landmark_mask"]
            landmark_timestamp = new_detected_timestamps

            if len(team_obs) != 0:
                _, action, _ = policy.low_level_act(
                    obs_tensor,
                    all_goals,
                    flatten_agent_major(prepared["low_rel_pos"]),
                    flatten_agent_major(prepared["low_masks"]),
                    deterministic=True,
                )
                actions.append(action.squeeze(1).cpu().numpy())

        return (
            np.hstack(actions),
            all_goals,
            all_tasks,
            landmark_data,
            landmark_mask,
            landmark_timestamp,
        )

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
