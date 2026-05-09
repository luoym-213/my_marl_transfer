import numpy as np
from scipy.ndimage import center_of_mass, label


def detect_targets(map_state):
    binary_map = (map_state.belief_grid > map_state.belief_threshold).astype(np.int8)

    structure = np.ones((3, 3), dtype=np.int8)
    cluster_labels, num_clusters = label(binary_map, structure=structure)

    target_positions = []
    target_grid_positions = []
    cluster_sizes = []

    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = cluster_labels == cluster_id
        size = np.sum(cluster_mask)
        cluster_sizes.append(size)

        grid_centroid = center_of_mass(cluster_mask)
        i_center, j_center = int(round(grid_centroid[0])), int(round(grid_centroid[1]))

        i_center = np.clip(i_center, 0, map_state.map_dim - 1)
        j_center = np.clip(j_center, 0, map_state.map_dim - 1)

        target_grid_positions.append((i_center, j_center))
        target_positions.append(map_state.grid_to_world((i_center, j_center)))

    return {
        "binary_map": binary_map,
        "num_targets": num_clusters,
        "target_positions": target_positions,
        "target_grid_positions": target_grid_positions,
        "cluster_sizes": cluster_sizes,
        "cluster_labels": cluster_labels,
    }


def get_target_positions(map_state, min_cluster_size=1):
    result = detect_targets(map_state)

    if min_cluster_size > 1:
        return [
            pos
            for pos, size in zip(result["target_positions"], result["cluster_sizes"])
            if size >= min_cluster_size
        ]

    return result["target_positions"]


def visualize_detected_targets(map_state):
    result = detect_targets(map_state)
    vis_map = result["cluster_labels"].astype(np.float32)

    for grid_pos in result["target_grid_positions"]:
        i, j = grid_pos
        vis_map[i, j] = -1

    result["visualization_map"] = vis_map
    return result


def get_targets_summary(map_state):
    result = detect_targets(map_state)

    return {
        "num_targets": result["num_targets"],
        "total_high_belief_cells": np.sum(result["binary_map"]),
        "mean_cluster_size": np.mean(result["cluster_sizes"])
        if result["cluster_sizes"]
        else 0,
        "max_cluster_size": max(result["cluster_sizes"])
        if result["cluster_sizes"]
        else 0,
        "min_cluster_size": min(result["cluster_sizes"])
        if result["cluster_sizes"]
        else 0,
        "target_positions": result["target_positions"],
    }
