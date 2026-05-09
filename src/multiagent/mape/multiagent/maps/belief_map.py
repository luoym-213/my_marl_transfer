import numpy as np

from . import exploration_rewards, grid, heatmap, target_detection, voronoi


class GlobalBeliefMap:
    """Facade for belief state, map geometry, rewards, and map-derived features."""

    def __init__(
        self,
        world_size=2.0,
        cell_size=0.02,
        initial_belief=0.5,
        sensor_fidelity=0.8,
        landmark_positions=None,
        landmark_radius=0.05,
        obs_radius=0.3,
    ):
        self.world_size = world_size
        self.cell_size = cell_size
        self.initial_belief = initial_belief
        self.sensor_fidelity = sensor_fidelity
        self.landmark_positions = landmark_positions if landmark_positions is not None else []
        self.landmark_radius = landmark_radius

        self.map_dim = int(world_size / cell_size)
        self.belief_grid = np.full(
            (self.map_dim, self.map_dim), initial_belief, dtype=np.float32
        )

        self.world_min = -world_size / 2.0
        self.world_max = world_size / 2.0
        self._precompute_cell_centers()

        self.landmark_map = np.zeros((self.map_dim, self.map_dim), dtype=bool)
        for lx, ly in self.landmark_positions:
            dist_sq = (self.cell_world_x - lx) ** 2 + (self.cell_world_y - ly) ** 2
            self.landmark_map |= dist_sq <= landmark_radius ** 2

        self.epsilon = 1e-10
        self.landmark_heatmap = self.get_landmarks_heatmap()
        self.belief_threshold = 0.95

        dummy_mask = self.get_fov_mask((0, 0), obs_radius)
        fov_area = np.sum(dummy_mask)

        p_s = self.sensor_fidelity
        max_drop = 0.0
        curr_p = self.initial_belief
        for _ in range(10):
            h_curr = -(curr_p * np.log2(curr_p) + (1 - curr_p) * np.log2(1 - curr_p))
            num = p_s * curr_p
            den = p_s * curr_p + (1 - p_s) * (1 - curr_p)
            next_p = num / den
            h_next = -(next_p * np.log2(next_p) + (1 - next_p) * np.log2(1 - next_p))

            drop = h_curr - h_next
            if drop > max_drop:
                max_drop = drop
            curr_p = next_p

        max_possible_entropy = max_drop * fov_area
        self.explore_reward_normalization = (
            max_possible_entropy if max_possible_entropy > 0 else 1.0
        )

        self.max_belief_delta = 0.0
        for b in np.linspace(0, 1, 101):
            num = p_s * b
            den = p_s * b + (1 - p_s) * (1 - b)
            if den > 1e-10:
                new_b = num / den
                delta = new_b - b
                if delta > self.max_belief_delta:
                    self.max_belief_delta = delta

        self._update_discover_normalization()

    def _update_discover_normalization(self):
        total_landmark_cells = np.sum(self.landmark_map)
        normalization_area = total_landmark_cells if total_landmark_cells > 0 else 1.0

        max_possible_discover = self.max_belief_delta * normalization_area
        self.discover_reward_normalization = (
            max_possible_discover if max_possible_discover > 0 else 1.0
        )

    def _precompute_cell_centers(self):
        self.cell_world_x, self.cell_world_y = grid.precompute_cell_centers(
            self.map_dim, self.world_min, self.cell_size
        )

    def reset(self, landmark_positions=None):
        self.belief_grid.fill(self.initial_belief)

        if landmark_positions is not None:
            self.landmark_positions = landmark_positions
            self.landmark_map = np.zeros((self.map_dim, self.map_dim), dtype=bool)
            for lx, ly in self.landmark_positions:
                dist_sq = (self.cell_world_x - lx) ** 2 + (
                    self.cell_world_y - ly
                ) ** 2
                self.landmark_map |= dist_sq <= self.landmark_radius ** 2

    def world_to_grid(self, world_pos):
        return grid.world_to_grid(self, world_pos)

    def grid_to_world(self, grid_pos):
        return grid.grid_to_world(self, grid_pos)

    def get_fov_mask(self, agent_pos, obs_radius):
        return grid.get_fov_mask(self, agent_pos, obs_radius)

    def bayesian_update(self, positive_mask, negative_mask):
        if np.any(positive_mask):
            b_prev = self.belief_grid[positive_mask]
            p_s = self.sensor_fidelity

            numerator = p_s * b_prev
            denominator = p_s * b_prev + (1 - p_s) * (1 - b_prev)
            denominator = np.maximum(denominator, self.epsilon)

            b_new = numerator / denominator
            self.belief_grid[positive_mask] = np.clip(b_new, 0.0, 1.0)

        if np.any(negative_mask):
            b_prev = self.belief_grid[negative_mask]
            p_s = self.sensor_fidelity

            numerator = (1 - p_s) * b_prev
            denominator = (1 - p_s) * b_prev + p_s * (1 - b_prev)
            denominator = np.maximum(denominator, self.epsilon)

            b_new = numerator / denominator
            self.belief_grid[negative_mask] = np.clip(b_new, 0.0, 1.0)

    def update_beliefs(self, agent_positions, obs_radius):
        fov_mask = np.zeros((self.map_dim, self.map_dim), dtype=bool)
        for agent_pos in agent_positions:
            fov_mask |= self.get_fov_mask(agent_pos, obs_radius)

        positive_mask = fov_mask & self.landmark_map
        negative_mask = fov_mask & (~self.landmark_map)
        self.bayesian_update(positive_mask, negative_mask)

    def compute_shannon_entropy(self):
        b = self.belief_grid

        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = np.where(
                (b > self.epsilon) & (b < 1 - self.epsilon),
                -(b * np.log2(b) + (1 - b) * np.log2(1 - b)),
                0.0,
            )

        return entropy

    def get_total_uncertainty(self):
        return np.sum(self.compute_shannon_entropy())

    def get_mean_uncertainty(self):
        return np.mean(self.compute_shannon_entropy())

    def get_high_uncertainty_positions(self, threshold=0.9):
        entropy = self.compute_shannon_entropy()
        high_uncertainty_mask = entropy >= threshold
        indices = np.argwhere(high_uncertainty_mask)

        return [(int(i), int(j)) for i, j in indices]

    def get_belief_at_position(self, world_pos):
        grid_pos = self.world_to_grid(world_pos)
        if grid_pos is None:
            return None

        i, j = grid_pos
        return self.belief_grid[i, j]

    def visualize_beliefs(self):
        return self.belief_grid.copy()

    def visualize_entropy(self):
        return self.compute_shannon_entropy()

    def compute_voronoi_regions(self, agent_positions, agent_dones=None):
        return voronoi.compute_voronoi_regions(self, agent_positions, agent_dones)

    def get_voronoi_edges(self, agent_positions, agent_dones=None):
        return voronoi.get_voronoi_edges(self, agent_positions, agent_dones)

    def get_voronoi_region_masks(self, agent_positions, agents_dones=None):
        return voronoi.get_voronoi_region_masks(self, agent_positions, agents_dones)

    def compute_entropy_weighted_centroids(self, agent_positions):
        return voronoi.compute_entropy_weighted_centroids(self, agent_positions)

    def get_voronoi_region_stats(self, agent_positions):
        return voronoi.get_voronoi_region_stats(self, agent_positions)

    def detect_targets(self):
        return target_detection.detect_targets(self)

    def get_target_positions(self, min_cluster_size=1):
        return target_detection.get_target_positions(self, min_cluster_size)

    def visualize_detected_targets(self):
        return target_detection.visualize_detected_targets(self)

    def get_targets_summary(self):
        return target_detection.get_targets_summary(self)

    def get_agents_heatmap(self, agent_positions, radius, sigma=None, clip_outside=True):
        return heatmap.get_agents_heatmap(
            self, agent_positions, radius, sigma, clip_outside
        )

    def get_landmarks_heatmap(
        self,
        radius=None,
        sigma=None,
        clip_outside=True,
        landmark_positions=None,
    ):
        return heatmap.get_landmarks_heatmap(
            self, radius, sigma, clip_outside, landmark_positions
        )

    def get_agent_step_rewards(
        self,
        agent_positions,
        radius,
        discover_reward_scale=1.0,
    ):
        return exploration_rewards.get_agent_step_rewards(
            self, agent_positions, radius, discover_reward_scale
        )

    def get_agent_step_explore_entropy(
        self,
        agent_positions,
        radius,
        sigma=None,
        clip_outside=True,
    ):
        return exploration_rewards.get_agent_step_explore_entropy(
            self, agent_positions, radius, sigma, clip_outside
        )

    def get_agent_discover_target_reward(
        self,
        agent_positions,
        radius,
        reward_value=1.0,
    ):
        return exploration_rewards.get_agent_discover_target_reward(
            self, agent_positions, radius, reward_value
        )
