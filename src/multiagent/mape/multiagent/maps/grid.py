import numpy as np


def precompute_cell_centers(map_dim, world_min, cell_size):
    grid_x, grid_y = np.meshgrid(
        np.arange(map_dim),
        np.arange(map_dim),
        indexing="ij",
    )
    cell_world_x = world_min + (grid_x + 0.5) * cell_size
    cell_world_y = world_min + (grid_y + 0.5) * cell_size
    return cell_world_x, cell_world_y


def world_to_grid(map_state, world_pos):
    x, y = world_pos

    if not (
        map_state.world_min <= x <= map_state.world_max
        and map_state.world_min <= y <= map_state.world_max
    ):
        return None

    i = int((x - map_state.world_min) / map_state.cell_size)
    j = int((y - map_state.world_min) / map_state.cell_size)

    i = np.clip(i, 0, map_state.map_dim - 1)
    j = np.clip(j, 0, map_state.map_dim - 1)

    return (i, j)


def grid_to_world(map_state, grid_pos):
    i, j = grid_pos
    x = map_state.world_min + (i + 0.5) * map_state.cell_size
    y = map_state.world_min + (j + 0.5) * map_state.cell_size
    return (x, y)


def get_fov_mask(map_state, agent_pos, obs_radius):
    x, y = agent_pos
    dist_sq = (map_state.cell_world_x - x) ** 2 + (map_state.cell_world_y - y) ** 2
    return dist_sq <= obs_radius ** 2
