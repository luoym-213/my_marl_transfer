import numpy as np
from scipy.spatial import Voronoi


def compute_voronoi_regions(map_state, agent_positions, agent_dones=None):
    if len(agent_positions) == 0:
        return None

    agent_grids = []
    for pos in agent_positions:
        grid_pos = map_state.world_to_grid(pos)
        if grid_pos is None:
            x, y = pos
            x = np.clip(x, map_state.world_min, map_state.world_max)
            y = np.clip(y, map_state.world_min, map_state.world_max)
            grid_pos = map_state.world_to_grid((x, y))
        agent_grids.append(grid_pos)

    agent_grids = np.array(agent_grids)
    grid_i, grid_j = np.meshgrid(
        np.arange(map_state.map_dim),
        np.arange(map_state.map_dim),
        indexing="ij",
    )

    voronoi_map = np.full((map_state.map_dim, map_state.map_dim), -1, dtype=np.int32)
    min_dist_map = np.full((map_state.map_dim, map_state.map_dim), np.inf)

    for agent_idx, (ai, aj) in enumerate(agent_grids):
        if agent_dones is not None and agent_dones[agent_idx]:
            continue

        dist_map = np.sqrt((grid_i - ai) ** 2 + (grid_j - aj) ** 2)
        mask = dist_map < min_dist_map
        voronoi_map[mask] = agent_idx
        min_dist_map[mask] = dist_map[mask]

    return voronoi_map


def get_voronoi_edges(map_state, agent_positions, agent_dones=None):
    if len(agent_positions) < 2:
        return []

    agent_positions = np.array(agent_positions)

    if agent_dones is not None:
        agent_dones = np.array(agent_dones)
        active_positions = agent_positions[~agent_dones]
        if len(active_positions) < 2:
            return []
    else:
        active_positions = agent_positions

    boundary = map_state.world_size / 2.0
    mirror_points = [
        [-boundary * 3, -boundary * 3],
        [-boundary * 3, boundary * 3],
        [boundary * 3, -boundary * 3],
        [boundary * 3, boundary * 3],
    ]

    for pos in active_positions:
        mirror_points.extend([
            [pos[0], boundary * 3],
            [pos[0], -boundary * 3],
            [boundary * 3, pos[1]],
            [-boundary * 3, pos[1]],
        ])

    all_points = np.vstack([active_positions, mirror_points])

    try:
        vor = Voronoi(all_points)
        edges = []

        for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in ridge_vertices:
                continue
            if not (
                ridge_points[0] < len(active_positions)
                or ridge_points[1] < len(active_positions)
            ):
                continue

            v0 = vor.vertices[ridge_vertices[0]]
            v1 = vor.vertices[ridge_vertices[1]]
            v0_clipped = np.clip(v0, -boundary, boundary)
            v1_clipped = np.clip(v1, -boundary, boundary)

            if (
                abs(v0_clipped[0]) <= boundary
                and abs(v0_clipped[1]) <= boundary
                and abs(v1_clipped[0]) <= boundary
                and abs(v1_clipped[1]) <= boundary
            ):
                edges.append((
                    (float(v0_clipped[0]), float(v0_clipped[1])),
                    (float(v1_clipped[0]), float(v1_clipped[1])),
                ))

        return edges
    except Exception:
        return []


def get_voronoi_region_masks(map_state, agent_positions, agents_dones=None):
    if len(agent_positions) == 0:
        return []

    voronoi_map = compute_voronoi_regions(map_state, agent_positions, agents_dones)
    if voronoi_map is None:
        return []

    masks = []
    for agent_idx in range(len(agent_positions)):
        if agents_dones is not None and agents_dones[agent_idx]:
            masks.append(np.zeros((map_state.map_dim, map_state.map_dim), dtype=bool))
        else:
            masks.append(voronoi_map == agent_idx)

    return masks


def compute_entropy_weighted_centroids(map_state, agent_positions):
    voronoi_map = compute_voronoi_regions(map_state, agent_positions)
    if voronoi_map is None:
        return []

    entropy_map = map_state.compute_shannon_entropy()
    centroids = []

    for agent_idx in range(len(agent_positions)):
        region_mask = voronoi_map == agent_idx
        weights = entropy_map * region_mask
        total_weight = np.sum(weights)

        if total_weight <= 0:
            centroids.append(tuple(agent_positions[agent_idx]))
            continue

        cx = np.sum(map_state.cell_world_x * weights) / total_weight
        cy = np.sum(map_state.cell_world_y * weights) / total_weight
        centroids.append((float(cx), float(cy)))

    return centroids


def get_voronoi_region_stats(map_state, agent_positions):
    voronoi_map = compute_voronoi_regions(map_state, agent_positions)
    entropy_map = map_state.compute_shannon_entropy()
    centroids = compute_entropy_weighted_centroids(map_state, agent_positions)

    stats = []
    for agent_idx in range(len(agent_positions)):
        region_mask = voronoi_map == agent_idx

        area = np.sum(region_mask)
        total_entropy = np.sum(entropy_map[region_mask])
        mean_entropy = np.mean(entropy_map[region_mask]) if area > 0 else 0.0
        mean_belief = np.mean(map_state.belief_grid[region_mask]) if area > 0 else 0.5

        stats.append({
            "agent_idx": agent_idx,
            "area": int(area),
            "total_entropy": float(total_entropy),
            "mean_entropy": float(mean_entropy),
            "centroid": centroids[agent_idx],
            "mean_belief": float(mean_belief),
        })

    return stats
