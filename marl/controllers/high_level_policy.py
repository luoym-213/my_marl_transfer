import numpy as np
import torch

from marl.controllers.planning.rrt_GNN import plan_batch


def build_explore_nodes(
    top_k,
    rrt_max_iter,
    vec_inp,
    map_inp,
    agent_indices=None,
):
    """
    Generate explore-node candidates with RRT planning.

    vec_inp: [Batch, num_agents, 4], world coords [x, y, goal_x, goal_y]
    map_inp: [2, Batch, num_agents, H, W], 0 entropy, 1 voronoi mask
    agent_indices: [N], optional subset of agents to update
    """
    batch_processes = vec_inp.size(0)
    if agent_indices is not None:
        batch_idx = torch.arange(batch_processes, device=vec_inp.device)
        update_nodes = vec_inp[batch_idx, agent_indices].unsqueeze(1)
        voronoi_np = (
            map_inp[1, batch_idx, agent_indices]
            .unsqueeze(1)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        entropy_np = (
            map_inp[0, batch_idx, agent_indices]
            .unsqueeze(1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        update_nodes = vec_inp
        voronoi_np = map_inp[1].detach().cpu().numpy().astype(bool)
        entropy_np = map_inp[0].detach().cpu().numpy().astype(np.float32)

    batch_agents = update_nodes.size(1)
    start_nodes = world_to_grid_torch(
        update_nodes.view(-1, update_nodes.size(2))[:, :2],
        H=100,
        W=100,
    )
    voronoi_inp = voronoi_np.reshape(-1, voronoi_np.shape[-2], voronoi_np.shape[-1])
    entropy_inp = entropy_np.reshape(-1, entropy_np.shape[-2], entropy_np.shape[-1])

    batch_rrt = plan_batch(
        start_nodes,
        voronoi_inp,
        entropy_inp,
        max_iterations=rrt_max_iter,
        top_k=top_k,
    )
    batch_rrt = torch.tensor(
        batch_rrt,
        dtype=torch.float32,
        device=vec_inp.device,
    ).view(batch_processes, batch_agents, -1, 3)

    explore_nodes_world = grid_to_world_torch(batch_rrt[..., :2], H=100, W=100)
    ego_positions = update_nodes[..., :2]
    relative_explore_positions = (
        explore_nodes_world - ego_positions.unsqueeze(2)
    )
    explore_nodes = torch.cat(
        [relative_explore_positions, batch_rrt[..., 2:3]],
        dim=-1,
    )

    distance_threshold = 0.3
    all_goals = vec_inp[..., 2:4]
    dists = torch.norm(
        explore_nodes_world.unsqueeze(3)
        - all_goals.unsqueeze(1).unsqueeze(1),
        dim=-1,
    )

    mask = torch.ones_like(dists, dtype=torch.bool)
    if agent_indices is not None:
        batch_idx = torch.arange(batch_processes, device=vec_inp.device)
        mask[batch_idx, 0, :, agent_indices] = False
    else:
        diag_mask = torch.eye(batch_agents, device=vec_inp.device).bool()
        mask = ~diag_mask.view(1, batch_agents, 1, batch_agents).expand(
            batch_processes,
            batch_agents,
            top_k,
            batch_agents,
        )

    dists = torch.where(
        mask,
        dists,
        torch.tensor(float("inf"), device=dists.device),
    )
    valid_mask = dists < distance_threshold
    occupied_values = ((distance_threshold - dists) / distance_threshold).pow(2)
    occupied_values = torch.where(
        valid_mask,
        occupied_values,
        torch.tensor(0.0, device=dists.device),
    )
    occupied_feature = occupied_values.sum(dim=-1, keepdim=True)

    return torch.cat([explore_nodes, occupied_feature], dim=-1)


def build_landmark_nodes(
    agent_positions,
    detected,
    detected_mask,
    linear_indices,
    num_agents,
    all_masks=None,
):
    batch_landmark_data = detected[linear_indices]
    batch_landmark_mask = detected_mask[linear_indices]
    ego_positions = agent_positions[linear_indices]

    relative_pos = batch_landmark_data[:, :, :2] - ego_positions.unsqueeze(1)
    batch_landmark_data_relative = torch.cat([
        relative_pos,
        batch_landmark_data[:, :, 2:],
    ], dim=2)

    if all_masks is not None:
        num_processes = all_masks.size(0) // num_agents
        env_indices = linear_indices % num_processes
        all_masks_reshaped = all_masks.view(num_agents, num_processes).t()
        retired_counts_per_env = (all_masks_reshaped < 0.5).sum(dim=1)
        num_retired_agents = retired_counts_per_env[env_indices]

        is_targeted = batch_landmark_data_relative[:, :, 3]
        valid_landmark_mask = batch_landmark_mask.squeeze(-1) > 0.5
        untargeted_and_valid = (is_targeted < 0.5) & valid_landmark_mask
        vals = (num_retired_agents + 2).float().unsqueeze(1)
        batch_landmark_data_relative[:, :, 2] = torch.where(
            untargeted_and_valid,
            vals,
            batch_landmark_data_relative[:, :, 2],
        )

    return batch_landmark_data_relative, batch_landmark_mask


def build_edge_features(
    explore_nodes,
    landmark_nodes,
    landmark_node_masks,
    norm=False,
    max_distance=2.8,
):
    explore_relative_pos = explore_nodes[:, :, :2]
    distances = torch.norm(explore_relative_pos, dim=2, keepdim=True)

    if norm:
        d_feature = distances / max_distance
    else:
        d_feature = distances

    distances_safe = distances.clamp(min=1e-6)
    cos_theta = explore_relative_pos[:, :, 0:1] / distances_safe
    sin_theta = explore_relative_pos[:, :, 1:2] / distances_safe
    batch_ego_to_explore_edges = torch.cat(
        [d_feature, cos_theta, sin_theta], dim=2
    )

    landmark_relative_pos = landmark_nodes[:, :, :2]
    distances_landmark = torch.norm(
        landmark_relative_pos, dim=2, keepdim=True
    )

    if norm:
        d_feature_landmark = distances_landmark / max_distance
    else:
        d_feature_landmark = distances_landmark

    distances_landmark_safe = distances_landmark.clamp(min=1e-6)
    cos_theta_landmark = (
        landmark_relative_pos[:, :, 0:1] / distances_landmark_safe
    )
    sin_theta_landmark = (
        landmark_relative_pos[:, :, 1:2] / distances_landmark_safe
    )
    batch_ego_to_landmark_edges = torch.cat(
        [d_feature_landmark, cos_theta_landmark, sin_theta_landmark],
        dim=2,
    )
    batch_ego_to_landmark_edge_masks = landmark_node_masks

    return (
        batch_ego_to_explore_edges,
        batch_ego_to_landmark_edges,
        batch_ego_to_landmark_edge_masks,
    )


def world_to_grid_torch(world_xy, H, W):
    world_min = -1.0
    cell_size_x = 2.0 / float(H)
    cell_size_y = 2.0 / float(W)

    x = world_xy[..., 0]
    y = world_xy[..., 1]

    i = torch.floor((x - world_min) / cell_size_x)
    j = torch.floor((y - world_min) / cell_size_y)

    i = i.clamp(0, H - 1)
    j = j.clamp(0, W - 1)

    return torch.stack([i, j], dim=-1).long()


def grid_to_world_torch(grid_ij, H, W):
    world_min = -1.0
    cell_size_x = 2.0 / float(H)
    cell_size_y = 2.0 / float(W)

    i = grid_ij[..., 0].float()
    j = grid_ij[..., 1].float()

    x = world_min + (i + 0.5) * cell_size_x
    y = world_min + (j + 0.5) * cell_size_y

    return torch.stack([x, y], dim=-1)


_world_to_grid_torch = world_to_grid_torch
_grid_to_world_torch = grid_to_world_torch


class HighLevelPolicy:
    """Runs the high-level goal selection path for a batch of agents."""

    def __init__(self, top_k, rrt_max_iter, device):
        self.top_k = top_k
        self.rrt_max_iter = rrt_max_iter
        self.device = device

    def select_goals(
        self,
        policy,
        obs_positions,
        all_masks,
        goals,
        tasks,
        goal_log_probs,
        goal_done_mask,
        agent_entropy_maps,
        voronoi_masks,
        agent_nodes,
        ego_nodes,
        teammate_nodes,
        teammate_mask,
        landmark_data,
        landmark_mask,
        num_processes,
        deterministic=False,
        update_tasks=True,
        update_log_probs=True,
        update_targeted=True,
        store_landmark_nodes=True,
        landmark_nodes_out=None,
    ):
        if not goal_done_mask.any():
            return {
                "has_decision": False,
                "goals": goals,
                "tasks": tasks,
                "goal_log_probs": goal_log_probs,
                "landmark_data": landmark_data,
                "landmark_mask": landmark_mask,
                "landmark_nodes": landmark_nodes_out,
            }

        update_indices = torch.nonzero(goal_done_mask, as_tuple=False)
        proc_indices = update_indices[:, 0]
        agent_indices = update_indices[:, 1]
        linear_indices = agent_indices * num_processes + proc_indices

        map_inputs = torch.stack([
            agent_entropy_maps[proc_indices],
            voronoi_masks[proc_indices],
        ], dim=0)
        agent_vec_inputs = agent_nodes[proc_indices]
        batch_teammate_nodes = teammate_nodes[proc_indices]
        batch_teammate_masks = teammate_mask[proc_indices].clone()

        batch_indices = torch.arange(len(proc_indices), device=self.device)
        batch_teammate_masks[batch_indices, agent_indices, 0] = 0.0

        batch_explore_nodes = build_explore_nodes(
            self.top_k,
            self.rrt_max_iter,
            agent_vec_inputs,
            map_inputs,
            agent_indices,
        )
        batch_explore_nodes = batch_explore_nodes.reshape(
            -1, batch_explore_nodes.shape[-2], batch_explore_nodes.shape[-1]
        )

        batch_ego_nodes = ego_nodes[proc_indices, agent_indices]
        batch_landmark_nodes, batch_landmark_node_masks = build_landmark_nodes(
            obs_positions,
            landmark_data,
            landmark_mask,
            linear_indices,
            num_agents=agent_nodes.size(1),
            all_masks=all_masks,
        )

        if store_landmark_nodes and landmark_nodes_out is not None:
            landmark_nodes_out[linear_indices] = batch_landmark_nodes

        edge_outputs = build_edge_features(
            batch_explore_nodes,
            batch_landmark_nodes,
            batch_landmark_node_masks,
        )
        (
            batch_ego_to_explore_edges,
            batch_ego_to_landmark_edges,
            batch_ego_to_landmark_edge_masks,
        ) = edge_outputs

        batch_goals = policy.get_high_level_goal(
            batch_ego_nodes,
            batch_teammate_nodes,
            batch_teammate_masks,
            batch_explore_nodes,
            batch_ego_to_explore_edges,
            batch_landmark_nodes,
            batch_landmark_node_masks,
            batch_ego_to_landmark_edges,
            batch_ego_to_landmark_edge_masks,
            deterministic=deterministic,
        )

        goals[linear_indices] = batch_goals["waypoints"]
        if update_tasks:
            tasks[linear_indices] = batch_goals["action_modes"]
        if update_log_probs and goal_log_probs is not None:
            goal_log_probs[linear_indices] = batch_goals["node_log_probs"]
        if update_targeted:
            self._mark_targeted_landmarks(
                landmark_data,
                landmark_mask,
                batch_goals,
                linear_indices,
                num_processes,
            )

        return {
            "has_decision": True,
            "goals": goals,
            "tasks": tasks,
            "goal_log_probs": goal_log_probs,
            "landmark_data": landmark_data,
            "landmark_mask": landmark_mask,
            "landmark_nodes": landmark_nodes_out,
            "linear_indices": linear_indices,
            "explore_nodes": batch_explore_nodes,
            "teammate_masks": batch_teammate_masks,
            "batch_goals": batch_goals,
        }

    def _mark_targeted_landmarks(
        self,
        landmark_data,
        landmark_mask,
        batch_goals,
        linear_indices,
        num_processes,
    ):
        for i, linear_idx in enumerate(linear_indices):
            if batch_goals["action_modes"][i, 0] != 1:
                continue

            selected_waypoint = batch_goals["waypoints"][i]
            landmarks = landmark_data[linear_idx]
            current_mask = landmark_mask[linear_idx]
            valid_mask = current_mask[:, 0] > 0.5
            if not valid_mask.any():
                continue

            landmark_positions = landmarks[:, :2]
            distances = torch.norm(landmark_positions - selected_waypoint, dim=1)
            distances = distances.masked_fill(~valid_mask, float("inf"))
            min_idx = distances.argmin()
            if distances[min_idx] < 0.05:
                landmark_data[:, min_idx, 3] = 1.0
                current_proc_idx = linear_idx % num_processes
                process_agent_indices = torch.arange(
                    current_proc_idx,
                    landmark_data.shape[0],
                    num_processes,
                    device=self.device,
                )
                landmark_data[process_agent_indices, min_idx, 3] = 1.0
