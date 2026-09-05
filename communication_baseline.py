"""Evaluation-time communication degradation and stale-state baseline.

This module deliberately leaves the trained HIG-SAR policy unchanged.  It
provides an environment with per-UAV local information and an evaluation action
path that consumes only the querying UAV's local view.
"""

from __future__ import division

import numpy as np
import torch

from learner import Learner
from multiagent.communication import CommunicationSimulator
from multiagent.environment import MultiAgentEnv
from multiagent.global_belief_map import GlobalBeliefMap
import multiagent.scenarios as scenarios


def _as_position(value, fallback):
    if value is None:
        return np.asarray(fallback, dtype=np.float32).copy()
    return np.asarray(value, dtype=np.float32).copy()


class EpisodeSeededEnv(object):
    """Reset an environment from an episode-indexed random seed.

    RRT candidate generation and the environment currently share NumPy's
    process-wide RNG. Re-seeding immediately before every reset prevents a
    condition-dependent trajectory from changing the next episode's initial
    scene. Communication uses its own RNG and is seeded independently.
    """

    def __init__(
            self, env, trajectory_seed=None, communication_seed=None):
        self.env = env
        self.trajectory_seed = trajectory_seed
        self.communication_seed = communication_seed
        self._episode_index = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed=None):
        self.trajectory_seed = seed
        self._episode_index = 0
        return seed

    def reset(self):
        episode_index = self._episode_index
        if self.trajectory_seed is not None:
            self.env.seed(int(self.trajectory_seed) + episode_index)
        if self.communication_seed is not None:
            communication = getattr(self.env, "communication", None)
            if communication is None:
                raise RuntimeError(
                    "communication_seed requires a communication environment"
                )
            communication.seed(int(self.communication_seed) + episode_index)
        result = self.env.reset()
        self._episode_index += 1
        return result

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)


class CommunicationDegradedEnv(MultiAgentEnv):
    """Multi-agent environment in which every UAV owns a local information set."""

    def __init__(self, *args, **kwargs):
        self.communication_level = kwargs.pop("communication_level", "mild")
        self.communication_seed = kwargs.pop("communication_seed", None)
        self._local_ready = False
        self.communication = None
        super(CommunicationDegradedEnv, self).__init__(*args, **kwargs)

        self.communication = CommunicationSimulator(
            self.n,
            level=self.communication_level,
            seed=self.communication_seed,
        )
        self.local_belief_maps = [self._new_local_belief_map() for _ in range(self.n)]
        self._initialize_local_information()

    def _new_local_belief_map(self):
        return GlobalBeliefMap(
            world_size=self.world_size,
            cell_size=self.cell_size,
            landmark_positions=self.landmark_positions,
            landmark_radius=0.05,
            obs_radius=self.world.mask_obs_dist,
        )

    def seed(self, seed=None):
        result = super(CommunicationDegradedEnv, self).seed(seed)
        if self.communication is not None:
            comm_seed = self.communication_seed
            self.communication.seed(seed if comm_seed is None else comm_seed)
        return result

    def _true_motion_states(self):
        return np.asarray([
            np.concatenate([agent.state.p_pos, agent.state.p_vel])
            for agent in self.agents
        ], dtype=np.float32)

    def _current_goals(self):
        return np.asarray([
            _as_position(agent.state.g_pos, agent.state.p_pos)
            for agent in self.agents
        ], dtype=np.float32)

    def _initialize_local_information(self):
        """Perform the chapter's mandatory t=0 synchronization."""
        states = self._true_motion_states()
        # ``simple_spread.reset_world`` does not clear AgentState.g_pos. A
        # previous episode's option must never enter the new local caches.
        goals = states[:, :2].copy()
        for agent, goal in zip(self.agents, goals):
            agent.state.g_pos = goal.copy()
        self.local_agent_states = np.repeat(states[None, :, :], self.n, axis=0)
        self.local_agent_goals = np.repeat(goals[None, :, :], self.n, axis=0)
        self.local_agent_tasks = np.zeros((self.n, self.n, 1), dtype=np.int64)
        active = (~np.asarray(self.agents_done, dtype=bool)).astype(np.float32)
        self.local_agent_active = np.repeat(active[None, :], self.n, axis=0)
        self.local_source_steps = np.zeros((self.n, self.n), dtype=np.int64)
        self.communication.reset()
        self._local_ready = True

    @staticmethod
    def _merge_belief_grid(local_map, received_grid):
        received = np.asarray(received_grid, dtype=local_map.belief_grid.dtype)
        if received.shape != local_map.belief_grid.shape:
            raise ValueError(
                "belief grid shape mismatch: {} != {}".format(
                    received.shape, local_map.belief_grid.shape
                )
            )
        prior = float(local_map.initial_belief)
        replace = (
            np.abs(received - prior)
            > np.abs(local_map.belief_grid - prior)
        )
        local_map.belief_grid[replace] = received[replace]

    def _apply_deliveries(self, query_step):
        for event in self.communication.deliver(query_step):
            receiver = event["receiver"]
            sender = event["sender"]
            source_step = event["source_step"]
            if source_step <= self.local_source_steps[receiver, sender]:
                continue
            payload = event["payload"]
            self.local_agent_states[receiver, sender] = payload["motion_state"]
            self.local_agent_goals[receiver, sender] = payload["goal"]
            self.local_agent_tasks[receiver, sender, 0] = payload["task"]
            self.local_agent_active[receiver, sender] = payload["active"]
            self.local_source_steps[receiver, sender] = source_step
            self._merge_belief_grid(
                self.local_belief_maps[receiver], payload["belief_grid"]
            )

    def _refresh_own_information(self, tasks):
        states = self._true_motion_states()
        goals = self._current_goals()
        active = (~np.asarray(self.agents_done, dtype=bool)).astype(np.float32)
        step = int(self.world.steps)
        for ego in range(self.n):
            self.local_agent_states[ego, ego] = states[ego]
            self.local_agent_goals[ego, ego] = goals[ego]
            self.local_agent_tasks[ego, ego, 0] = int(tasks[ego])
            self.local_agent_active[ego, ego] = active[ego]
            self.local_source_steps[ego, ego] = step

    def _make_payloads(self, tasks):
        states = self._true_motion_states()
        goals = self._current_goals()
        active = (~np.asarray(self.agents_done, dtype=bool)).astype(np.float32)
        return [
            {
                "motion_state": states[sender].copy(),
                "goal": goals[sender].copy(),
                "task": int(tasks[sender]),
                "active": float(active[sender]),
                # One immutable snapshot is shared by all receiver queue events.
                "belief_grid": self.local_belief_maps[sender].belief_grid.copy(),
            }
            for sender in range(self.n)
        ]

    def _advance_local_information(self, tasks):
        query_step = int(self.world.steps)

        # First accept packets generated at earlier source times.  Their map
        # knowledge may consequently be relayed by a packet generated now.
        self._apply_deliveries(query_step)

        positions = np.asarray(
            [agent.state.p_pos for agent in self.agents], dtype=np.float32
        )
        for ego, local_map in enumerate(self.local_belief_maps):
            local_map.update_beliefs(
                positions[ego:ego + 1], self.world.mask_obs_dist
            )
        self._refresh_own_information(tasks)

        self.communication.schedule_broadcasts(
            query_step, positions, self._make_payloads(tasks)
        )
        self._apply_deliveries(query_step)

    def _local_target_positions(self):
        return [
            np.asarray(local_map.get_target_positions(), dtype=np.float32).reshape(-1, 2)
            for local_map in self.local_belief_maps
        ]

    def _augment_local_info(self, info):
        entropy_maps = []
        voronoi_masks = []
        heatmaps = []
        for ego, local_map in enumerate(self.local_belief_maps):
            entropy_maps.append(local_map.compute_shannon_entropy())
            local_positions = self.local_agent_states[ego, :, :2]
            local_dones = self.local_agent_active[ego] < 0.5
            all_masks = local_map.get_voronoi_region_masks(
                local_positions, local_dones
            )
            voronoi_masks.append(all_masks[ego])
            heatmaps.append(local_map.get_agents_heatmap(local_positions, 0.05))

        # These base-environment fields contain the synchronized global map or
        # exact landmark heatmap. They are intentionally removed so a future
        # evaluator cannot accidentally bypass the ego-local information path.
        for key in ("map", "entropy_map", "voronoi_masks", "heatmap"):
            info.pop(key, None)

        query_step = int(self.world.steps)
        if query_step == 0:
            # Learner.set_envs_info requires a reset-time heatmap to initialize
            # its legacy cache. The stale evaluator never consumes the cache;
            # initialize it with no environment truth.
            info["landmark_heatmap"] = np.zeros_like(
                self.local_belief_maps[0].landmark_heatmap,
                dtype=np.float32,
            )
        else:
            info.pop("landmark_heatmap", None)
        info["local_entropy_maps"] = np.stack(entropy_maps).astype(np.float32)
        info["local_voronoi_masks"] = np.stack(voronoi_masks).astype(bool)
        info["local_agent_heatmaps"] = np.stack(heatmaps).astype(np.float32)
        info["local_target_positions"] = self._local_target_positions()
        info["local_agent_states"] = self.local_agent_states.copy()
        info["local_agent_goals"] = self.local_agent_goals.copy()
        info["local_agent_tasks"] = self.local_agent_tasks.copy()
        info["local_agent_active"] = self.local_agent_active.copy()
        info["local_source_steps"] = self.local_source_steps.copy()
        info["local_aoi"] = (
            query_step - self.local_source_steps
        ).astype(np.int64)
        info["communication_level"] = self.communication_level
        info["communication_stats"] = self.communication.get_stats()
        return info

    def _get_obs(self, agent):
        observation = np.asarray(
            super(CommunicationDegradedEnv, self)._get_obs(agent),
            dtype=np.float32,
        ).copy()
        if not self._local_ready:
            return observation

        ego = int(agent.iden)
        base_size = 4 + 2 * self.n + 2 * (self.n - 1)
        identity_size = observation.size - base_size
        if identity_size < 0:
            raise ValueError("Unexpected simple_spread observation layout")

        own_position = np.asarray(agent.state.p_pos, dtype=np.float32)
        landmark_start = identity_size + 4
        landmark_end = landmark_start + 2 * self.n
        local_landmarks = np.zeros((self.n, 2), dtype=np.float32)
        detected = self._local_target_positions()[ego]
        count = min(self.n, detected.shape[0])
        if count:
            local_landmarks[:count] = detected[:count] - own_position
        observation[landmark_start:landmark_end] = local_landmarks.reshape(-1)

        other_positions = [
            self.local_agent_states[ego, other, :2] - own_position
            for other in range(self.n) if other != ego
        ]
        observation[landmark_end:] = np.asarray(
            other_positions, dtype=np.float32
        ).reshape(-1)
        return observation

    def reset(self):
        _, state, info = super(CommunicationDegradedEnv, self).reset()
        for local_map in self.local_belief_maps:
            local_map.reset(self.landmark_positions)
        self._initialize_local_information()
        observations = [self._get_obs(agent) for agent in self.agents]
        return observations, state, self._augment_local_info(info)

    def step(self, data, goal_n=None):
        (
            _, reward, high_reward, done, info, state
        ) = super(CommunicationDegradedEnv, self).step(data, goal_n=goal_n)
        tasks = np.asarray(data["agents_tasks"]).reshape(self.n, -1)[:, 0]
        self._advance_local_information(tasks)
        observations = [self._get_obs(agent) for agent in self.agents]
        return (
            observations,
            reward,
            high_reward,
            done,
            self._augment_local_info(info),
            state,
        )


def make_communication_env(
        env_id, num_agents, dist_threshold, arena_size, identity_size,
        mask_obs_dist=0.5, communication_level="mild",
        communication_seed=None, trajectory_seed=None):
    scenario = scenarios.load(env_id + ".py").Scenario(
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
    )
    world = scenario.make_world()
    env = CommunicationDegradedEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info if hasattr(scenario, "info") else None,
        state_callback=scenario.state,
        discrete_action=True,
        done_callback=scenario.done,
        cam_range=arena_size,
        mask_obs_dist=mask_obs_dist,
        communication_level=communication_level,
        communication_seed=communication_seed,
    )
    if trajectory_seed is None:
        return env
    return EpisodeSeededEnv(
        env,
        trajectory_seed=trajectory_seed,
        communication_seed=communication_seed,
    )


def make_ideal_env(
        env_id, num_agents, dist_threshold, arena_size, identity_size,
        mask_obs_dist=0.5, trajectory_seed=None):
    """Build the ideal-communication control with fair episode resets."""
    scenario = scenarios.load(env_id + ".py").Scenario(
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
    )
    world = scenario.make_world()
    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info if hasattr(scenario, "info") else None,
        state_callback=scenario.state,
        discrete_action=True,
        done_callback=scenario.done,
        cam_range=arena_size,
        mask_obs_dist=mask_obs_dist,
    )
    return EpisodeSeededEnv(env, trajectory_seed=trajectory_seed)


def _apply_local_claims(
        landmark_data, landmark_mask, local_goals, local_tasks, local_active,
        match_threshold=0.05):
    """Reconstruct target claims from each ego UAV's stale task cache."""
    landmark_data[:, :, 3] = 0.0
    if landmark_data.size(1) == 0:
        return
    targets = landmark_data[:, :, :2]
    distances = torch.linalg.vector_norm(
        targets[:, :, None, :] - local_goals[:, None, :, :], dim=-1
    )
    collecting = local_tasks[:, :, 0] == 1
    active = local_active > 0.5
    claimed = (
        (distances < match_threshold)
        & collecting[:, None, :]
        & active[:, None, :]
    ).any(dim=-1)
    valid = landmark_mask[:, :, 0] > 0.5
    landmark_data[:, :, 3] = (claimed & valid).to(landmark_data.dtype)


def stale_eval_act(
        self, observations, env_states, masks, goals, tasks,
        landmark_data, landmark_mask, deterministic=True):
    """Drop-in ``Learner.eval_act`` using only each ego UAV's local cache."""
    if len(self.teams_list) != 1 or len(self.teams_list[0]) != len(observations):
        raise NotImplementedError(
            "The stale baseline currently supports the homogeneous UAV team"
        )
    team = self.teams_list[0]
    policy = self.policies_list[0]
    num_agents = len(team)
    obs_tensor = torch.as_tensor(
        np.asarray(observations), dtype=torch.float32, device=self.device
    )
    all_goals = goals
    all_tasks = tasks
    info = self.envs_info

    entropy_maps = np.asarray(info["local_entropy_maps"], dtype=np.float32)
    voronoi_masks = np.asarray(info["local_voronoi_masks"], dtype=bool)
    goal_done = np.asarray(info["goal_done"], dtype=bool)

    local_states = torch.as_tensor(
        info["local_agent_states"], dtype=torch.float32, device=self.device
    )
    local_goals = torch.as_tensor(
        info["local_agent_goals"], dtype=torch.float32, device=self.device
    )
    local_tasks = torch.as_tensor(
        info["local_agent_tasks"], dtype=torch.long, device=self.device
    )
    local_active = torch.as_tensor(
        info["local_agent_active"], dtype=torch.float32, device=self.device
    )

    detected_maps = info["local_target_positions"]
    new_detected, new_detected_masks = self.update_landmark_info(
        landmark_data,
        landmark_mask,
        detected_maps,
        self.device,
    )
    _apply_local_claims(
        new_detected, new_detected_masks,
        local_goals, local_tasks, local_active,
    )

    if bool(goal_done.any()):
        agent_indices_np = np.flatnonzero(goal_done)
        agent_indices = torch.as_tensor(
            agent_indices_np, dtype=torch.long, device=self.device
        )
        batch_indices = torch.arange(
            agent_indices.numel(), device=self.device
        )

        batch_states = local_states[agent_indices]
        batch_goals_view = local_goals[agent_indices]
        vec_inp_agents = torch.cat([
            batch_states[:, :, :2], batch_goals_view
        ], dim=-1)

        teammate_distance_to_goal = torch.linalg.vector_norm(
            batch_goals_view - batch_states[:, :, :2],
            dim=-1,
            keepdim=True,
        )
        batch_teammate_nodes = torch.cat([
            batch_states,
            teammate_distance_to_goal,
        ], dim=-1)
        batch_teammate_masks = local_active[agent_indices].unsqueeze(-1).clone()
        batch_teammate_masks[batch_indices, agent_indices, 0] = 0.0

        batch_explore_nodes = policy.get_explore_nodes(
            self.top_k,
            self.rrt_max_iter,
            vec_inp_agents,
            entropy_maps[agent_indices_np],
            voronoi_masks[agent_indices_np],
            agent_indices,
        ).reshape(-1, self.top_k, 4)

        battery = (50.0 - float(info["world_steps"])) / 50.0
        ego_motion = batch_states[batch_indices, agent_indices]
        batch_ego_nodes = torch.cat([
            ego_motion,
            torch.full(
                (agent_indices.numel(), 1), battery,
                dtype=torch.float32, device=self.device,
            ),
        ], dim=-1)

        ego_positions = local_states[
            torch.arange(num_agents, device=self.device),
            torch.arange(num_agents, device=self.device),
            :2,
        ]
        batch_landmark_nodes, batch_landmark_masks = policy.get_landmark_nodes(
            ego_positions,
            new_detected,
            new_detected_masks,
            agent_indices,
        )
        (
            explore_edges,
            landmark_edges,
            landmark_edge_masks,
        ) = policy.get_edge_features(
            batch_explore_nodes,
            batch_landmark_nodes,
            batch_landmark_masks,
        )
        decision = policy.get_high_level_goal(
            batch_ego_nodes,
            batch_teammate_nodes,
            batch_teammate_masks,
            batch_explore_nodes,
            explore_edges,
            batch_landmark_nodes,
            batch_landmark_masks,
            landmark_edges,
            landmark_edge_masks,
            deterministic=deterministic,
        )
        all_goals[agent_indices] = decision["waypoints"]
        all_tasks[agent_indices] = decision["action_modes"]

        selected_data = new_detected[agent_indices]
        selected_valid = new_detected_masks[agent_indices, :, 0] > 0.5
        selected_distances = torch.linalg.vector_norm(
            selected_data[:, :, :2] - decision["waypoints"][:, None, :],
            dim=-1,
        ).masked_fill(~selected_valid, float("inf"))
        minimum_distance, selected_slots = selected_distances.min(dim=1)
        selected_rows = (
            (decision["action_modes"][:, 0] == 1)
            & (minimum_distance < 0.05)
        )
        selected_agents = agent_indices[selected_rows]
        selected_slots = selected_slots[selected_rows]
        new_detected[selected_agents, selected_slots, 3] = 1.0

    _, action, _ = policy.low_level_act(
        obs_tensor, all_goals, deterministic=True
    )
    return (
        action.squeeze(1).cpu().numpy(),
        all_goals,
        all_tasks,
        new_detected,
        new_detected_masks,
    )


def install_stale_evaluator(learner):
    """Bind the local-information evaluation path to one learner instance."""
    import types
    learner.eval_act = types.MethodType(stale_eval_act, learner)
    return learner
