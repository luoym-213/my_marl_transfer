import numpy as np
import torch


def stack_global_maps(envs_info, device):
    entropy_maps = torch.stack([
        torch.from_numpy(np.array(info["entropy_map"])).float()
        for info in envs_info
    ]).to(device)
    heatmaps = torch.stack([
        torch.from_numpy(np.array(info["heatmap"])).float()
        for info in envs_info
    ]).to(device)
    landmark_heatmaps = torch.stack([
        torch.from_numpy(np.array(info["landmark_heatmap"])).float()
        for info in envs_info
    ]).to(device)
    critic_map_input = torch.stack(
        [entropy_maps, heatmaps, landmark_heatmaps], dim=1
    )
    return entropy_maps, heatmaps, landmark_heatmaps, critic_map_input


def goal_done_mask(envs_info, device):
    return torch.tensor(
        [info["goal_done"] for info in envs_info],
        dtype=torch.bool,
        device=device,
    )


def agent_batteries(envs_info, num_agents, device):
    world_steps = torch.tensor(
        [info["world_steps"] for info in envs_info],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1).repeat(1, num_agents).unsqueeze(-1)
    return (50.0 - world_steps) / 50.0


def detected_maps(envs_info, device):
    return [
        torch.from_numpy(np.array(info["map"][1])).float().to(device)
        for info in envs_info
    ]


def local_detections(envs_info):
    return [info.get("local_detections", []) for info in envs_info]


def agent_alive_masks(envs_info, num_agents, device):
    return torch.from_numpy(
        np.array([
            info.get("agent_alive_mask", np.ones(num_agents, dtype=np.float32))
            for info in envs_info
        ], dtype=np.float32)
    ).to(
        dtype=torch.float32,
        device=device,
    )


def voronoi_masks(envs_info, num_agents, device):
    return torch.stack([
        torch.stack([
            torch.from_numpy(np.array(info["voronoi_masks"][a])).float()
            for a in range(num_agents)
        ])
        for info in envs_info
    ]).to(device)


def build_agent_context(all_obs, all_goals, all_masks, batteries,
                        num_agents, num_processes):
    agent_positions = all_obs[:, 2:4].view(
        num_agents, num_processes, 2
    ).transpose(0, 1)
    agent_vels = all_obs[:, 0:2].view(
        num_agents, num_processes, 2
    ).transpose(0, 1)
    agent_goals = all_goals.view(
        num_agents, num_processes, 2
    ).transpose(0, 1)

    agent_nodes = torch.cat([agent_positions, agent_goals], dim=-1)
    ego_nodes = torch.cat([agent_positions, agent_vels, batteries], dim=-1)
    dist_to_goal = torch.norm(
        agent_goals - agent_positions, dim=-1, keepdim=True
    )
    teammate_nodes = torch.cat(
        [agent_positions, agent_vels, dist_to_goal], dim=-1
    )
    teammate_mask = all_masks.view(
        num_agents, num_processes
    ).t().unsqueeze(-1)

    return {
        "agent_positions": agent_positions,
        "agent_vels": agent_vels,
        "agent_goals": agent_goals,
        "agent_nodes": agent_nodes,
        "ego_nodes": ego_nodes,
        "teammate_nodes": teammate_nodes,
        "teammate_mask": teammate_mask,
    }
