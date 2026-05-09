import numpy as np


def get_agents_heatmap(map_state, agent_positions, radius, sigma=None, clip_outside=True):
    if len(agent_positions) == 0:
        return np.zeros((map_state.map_dim, map_state.map_dim), dtype=np.float32)

    if sigma is None:
        sigma = radius / 3.0

    heatmap = np.zeros((map_state.map_dim, map_state.map_dim), dtype=np.float32)

    for agent_pos in agent_positions:
        x, y = agent_pos
        dist_x = map_state.cell_world_x - x
        dist_y = map_state.cell_world_y - y
        distance = np.sqrt(dist_x ** 2 + dist_y ** 2)
        agent_heatmap = np.exp(-(distance ** 2) / (2 * sigma ** 2))

        if clip_outside:
            agent_heatmap[distance > radius] = 0.0

        heatmap = np.maximum(heatmap, agent_heatmap)

    return heatmap.astype(np.float32)


def get_landmarks_heatmap(
    map_state,
    radius=None,
    sigma=None,
    clip_outside=True,
    landmark_positions=None,
):
    if landmark_positions is None:
        landmark_positions = map_state.landmark_positions

    if len(landmark_positions) == 0:
        return np.zeros((map_state.map_dim, map_state.map_dim), dtype=np.float32)

    if radius is None:
        radius = map_state.landmark_radius

    if sigma is None:
        sigma = radius / 3.0

    heatmap = np.zeros((map_state.map_dim, map_state.map_dim), dtype=np.float32)

    for landmark_pos in landmark_positions:
        x, y = landmark_pos
        dist_x = map_state.cell_world_x - x
        dist_y = map_state.cell_world_y - y
        distance = np.sqrt(dist_x ** 2 + dist_y ** 2)
        landmark_heatmap = np.exp(-(distance ** 2) / (2 * sigma ** 2))

        if clip_outside:
            landmark_heatmap[distance > radius] = 0.0

        heatmap = np.maximum(heatmap, landmark_heatmap)

    return heatmap.astype(np.float32)
