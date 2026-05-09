import torch


def build_comm_mask(agent_positions, comm_dist, alive_mask=None):
    """Build communication visibility.

    agent_positions: [P, A, 2]
    alive_mask: optional [P, A], 1 means agent is active
    returns: bool [P, A, A], receiver i can read sender j when True
    """
    deltas = agent_positions.unsqueeze(2) - agent_positions.unsqueeze(1)
    dists = torch.norm(deltas, dim=-1)
    if comm_dist is None:
        comm_mask = torch.ones_like(dists, dtype=torch.bool)
    else:
        comm_mask = dists <= float(comm_dist)

    if alive_mask is not None:
        alive = alive_mask > 0.5
        comm_mask = comm_mask & alive.unsqueeze(2) & alive.unsqueeze(1)

    return comm_mask


def build_low_level_comm_features(agent_positions, comm_mask):
    """Build fixed low-level teammate features.

    agent_positions: [P, A, 2]
    comm_mask: [P, A, A], receiver-source mask including self
    returns:
        rel_pos: [P, A, A-1, 2], teammate_pos - self_pos or 0
        masks: [P, A, A-1], 1 for communication-visible teammate
    """
    num_processes, num_agents, _ = agent_positions.shape
    rel_pos = torch.zeros(
        num_processes,
        num_agents,
        max(num_agents - 1, 0),
        2,
        device=agent_positions.device,
        dtype=agent_positions.dtype,
    )
    masks = torch.zeros(
        num_processes,
        num_agents,
        max(num_agents - 1, 0),
        device=agent_positions.device,
        dtype=agent_positions.dtype,
    )

    for agent_idx in range(num_agents):
        teammate_indices = [
            idx for idx in range(num_agents) if idx != agent_idx
        ]
        if not teammate_indices:
            continue
        teammate_pos = agent_positions[:, teammate_indices]
        self_pos = agent_positions[:, agent_idx].unsqueeze(1)
        visible = comm_mask[:, agent_idx, teammate_indices].float()
        rel_pos[:, agent_idx] = (teammate_pos - self_pos) * visible.unsqueeze(-1)
        masks[:, agent_idx] = visible

    return rel_pos, masks


def flatten_agent_major(tensor):
    """Convert [P, A, ...] to [A*P, ...] to match rollout chunk order."""
    if tensor.dim() < 2:
        raise ValueError("Expected tensor with leading [P, A] dimensions")
    num_processes, num_agents = tensor.shape[:2]
    return tensor.transpose(0, 1).contiguous().view(
        num_processes * num_agents, *tensor.shape[2:]
    )


class LocalMapBank:
    """Per-agent local map state owned by learner/controller code.

    Main tensors:
        belief_maps: [P, A, H, W]
        timestamp_maps: [P, A, H, W]
        entropy_maps: [P, A, H, W] cache, recomputed from belief_maps
    """

    def __init__(
        self,
        num_processes,
        num_agents,
        device,
        height=100,
        width=100,
        world_size=2.0,
        initial_belief=0.5,
        initial_timestamp=-1.0,
        sensor_fidelity=0.8,
        landmark_radius=0.05,
    ):
        self.num_processes = num_processes
        self.num_agents = num_agents
        self.device = device
        self.height = height
        self.width = width
        self.world_size = world_size
        self.world_min = -world_size / 2.0
        self.world_max = world_size / 2.0
        self.cell_size_x = world_size / float(height)
        self.cell_size_y = world_size / float(width)
        self.initial_belief = initial_belief
        self.initial_timestamp = initial_timestamp
        self.sensor_fidelity = sensor_fidelity
        self.landmark_radius = landmark_radius
        self.epsilon = 1e-6

        self.belief_maps = torch.full(
            (num_processes, num_agents, height, width),
            initial_belief,
            dtype=torch.float32,
            device=device,
        )
        self.timestamp_maps = torch.full(
            (num_processes, num_agents, height, width),
            initial_timestamp,
            dtype=torch.float32,
            device=device,
        )
        self.entropy_maps = None

        grid_i, grid_j = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.cell_world_x = self.world_min + (grid_i + 0.5) * self.cell_size_x
        self.cell_world_y = self.world_min + (grid_j + 0.5) * self.cell_size_y

    def reset(self, env_indices=None):
        """Reset all maps or selected process indices.

        env_indices: None, bool [P], or integer indices.
        """
        if env_indices is None:
            self.belief_maps.fill_(self.initial_belief)
            self.timestamp_maps.fill_(self.initial_timestamp)
        else:
            if isinstance(env_indices, torch.Tensor) and env_indices.dtype == torch.bool:
                env_indices = torch.nonzero(env_indices, as_tuple=False).flatten()
            if isinstance(env_indices, torch.Tensor) and env_indices.numel() == 0:
                return
            self.belief_maps[env_indices].fill_(self.initial_belief)
            self.timestamp_maps[env_indices].fill_(self.initial_timestamp)
        self.entropy_maps = None

    def update_from_local_observations(
        self,
        agent_positions,
        local_detections,
        step_id,
        sensor_dist,
    ):
        """Apply local sensing updates.

        agent_positions: [P, A, 2]
        local_detections: list[P][A] with arrays/tensors [num_visible, 2]
        step_id: scalar or [P] tensor/list, written into updated FOV cells
        sensor_dist: scalar local observation radius
        """
        step_values = self._step_values(step_id, agent_positions.shape[0])
        for proc_idx in range(agent_positions.shape[0]):
            for agent_idx in range(agent_positions.shape[1]):
                pos = agent_positions[proc_idx, agent_idx]
                fov_mask = self.get_fov_mask(pos, sensor_dist)
                fov_mask = fov_mask & (
                    self.timestamp_maps[proc_idx, agent_idx]
                    != step_values[proc_idx]
                )
                if not fov_mask.any():
                    continue

                detections = self._detections_tensor(
                    local_detections, proc_idx, agent_idx
                )
                positive_mask = torch.zeros_like(fov_mask)
                if detections.numel() > 0:
                    dx = self.cell_world_x.unsqueeze(0) - detections[:, 0].view(-1, 1, 1)
                    dy = self.cell_world_y.unsqueeze(0) - detections[:, 1].view(-1, 1, 1)
                    detection_mask = (dx.square() + dy.square()).le(
                        self.landmark_radius ** 2
                    ).any(dim=0)
                    positive_mask = fov_mask & detection_mask

                negative_mask = fov_mask & (~positive_mask)
                self._bayesian_update(proc_idx, agent_idx, positive_mask, True)
                self._bayesian_update(proc_idx, agent_idx, negative_mask, False)
                self.timestamp_maps[proc_idx, agent_idx][fov_mask] = step_values[
                    proc_idx
                ]

        self.entropy_maps = None

    def fuse_by_timestamp(self, comm_mask):
        """Fuse communicating neighbors by per-cell latest timestamp.

        comm_mask: bool [P, A, A], receiver-source visibility including self.
        Ties keep the receiver value because the receiver is placed first in
        the source set before torch.argmax is applied.
        """
        comm_mask = comm_mask.bool()
        fused_beliefs = self.belief_maps.clone()
        fused_timestamps = self.timestamp_maps.clone()

        for proc_idx in range(self.num_processes):
            for receiver_idx in range(self.num_agents):
                neighbor_indices = torch.nonzero(
                    comm_mask[proc_idx, receiver_idx],
                    as_tuple=False,
                ).flatten()
                if neighbor_indices.numel() == 0:
                    neighbor_indices = torch.tensor(
                        [receiver_idx], device=self.device, dtype=torch.long
                    )

                ordered_sources = self._receiver_first_sources(
                    receiver_idx, neighbor_indices
                )
                source_timestamps = self.timestamp_maps[proc_idx, ordered_sources]
                source_beliefs = self.belief_maps[proc_idx, ordered_sources]
                winner_indices = source_timestamps.argmax(dim=0)
                fused_timestamps[proc_idx, receiver_idx] = torch.gather(
                    source_timestamps, 0, winner_indices.unsqueeze(0)
                ).squeeze(0)
                fused_beliefs[proc_idx, receiver_idx] = torch.gather(
                    source_beliefs, 0, winner_indices.unsqueeze(0)
                ).squeeze(0)

        self.belief_maps = fused_beliefs
        self.timestamp_maps = fused_timestamps
        self.entropy_maps = None
        return self.belief_maps, self.timestamp_maps

    def compute_entropy_maps(self):
        """Return entropy maps [P, A, H, W] recomputed from belief_maps."""
        b = self.belief_maps.clamp(self.epsilon, 1.0 - self.epsilon)
        self.entropy_maps = -(b * torch.log2(b) + (1.0 - b) * torch.log2(1.0 - b))
        return self.entropy_maps

    def get_agent_entropy_maps(self):
        if self.entropy_maps is None:
            return self.compute_entropy_maps()
        return self.entropy_maps

    def get_local_voronoi_masks(
        self,
        agent_positions,
        comm_mask,
        scope="comm",
        alive_mask=None,
    ):
        """Build local planning masks.

        agent_positions: [P, A, 2]
        comm_mask: [P, A, A]
        returns: bool [P, A, H, W], receiver-owned local Voronoi cells
        """
        masks = torch.zeros(
            self.num_processes,
            self.num_agents,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.bool,
        )
        grid_xy = torch.stack([self.cell_world_x, self.cell_world_y], dim=-1)
        alive_bool = None if alive_mask is None else alive_mask > 0.5

        for proc_idx in range(self.num_processes):
            for receiver_idx in range(self.num_agents):
                if alive_bool is not None and not alive_bool[proc_idx, receiver_idx]:
                    continue
                if scope == "self":
                    masks[proc_idx, receiver_idx].fill_(True)
                    continue

                source_indices = torch.nonzero(
                    comm_mask[proc_idx, receiver_idx],
                    as_tuple=False,
                ).flatten()
                if source_indices.numel() == 0:
                    source_indices = torch.tensor(
                        [receiver_idx], device=self.device, dtype=torch.long
                    )
                if not (source_indices == receiver_idx).any():
                    source_indices = torch.cat([
                        torch.tensor(
                            [receiver_idx], device=self.device, dtype=torch.long
                        ),
                        source_indices,
                    ])

                source_positions = agent_positions[proc_idx, source_indices]
                dist_sq = (
                    grid_xy.unsqueeze(0) - source_positions.view(-1, 1, 1, 2)
                ).square().sum(dim=-1)
                closest_source = dist_sq.argmin(dim=0)
                receiver_source_idx = torch.nonzero(
                    source_indices == receiver_idx, as_tuple=False
                ).flatten()[0]
                masks[proc_idx, receiver_idx] = closest_source == receiver_source_idx

        return masks

    def get_agents_heatmap(self, agent_positions, radius=0.05):
        """Return simple agent occupancy heatmaps [P, H, W] for critic input."""
        heatmaps = torch.zeros(
            self.num_processes,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.float32,
        )
        for proc_idx in range(agent_positions.shape[0]):
            proc_heatmap = torch.zeros_like(heatmaps[proc_idx], dtype=torch.bool)
            for agent_idx in range(agent_positions.shape[1]):
                proc_heatmap |= self.get_fov_mask(
                    agent_positions[proc_idx, agent_idx], radius
                )
            heatmaps[proc_idx] = proc_heatmap.float()
        return heatmaps

    def get_fov_mask(self, agent_pos, radius):
        dx = self.cell_world_x - agent_pos[0]
        dy = self.cell_world_y - agent_pos[1]
        return dx.square() + dy.square() <= float(radius) ** 2

    def world_to_grid(self, world_xy):
        x = world_xy[..., 0]
        y = world_xy[..., 1]
        i = torch.floor((x - self.world_min) / self.cell_size_x).long()
        j = torch.floor((y - self.world_min) / self.cell_size_y).long()
        i = i.clamp(0, self.height - 1)
        j = j.clamp(0, self.width - 1)
        return torch.stack([i, j], dim=-1)

    def grid_to_world(self, grid_ij):
        i = grid_ij[..., 0].float()
        j = grid_ij[..., 1].float()
        x = self.world_min + (i + 0.5) * self.cell_size_x
        y = self.world_min + (j + 0.5) * self.cell_size_y
        return torch.stack([x, y], dim=-1)

    def _bayesian_update(self, proc_idx, agent_idx, mask, positive):
        if not mask.any():
            return

        b_prev = self.belief_maps[proc_idx, agent_idx][mask]
        p_s = self.sensor_fidelity
        if positive:
            numerator = p_s * b_prev
            denominator = p_s * b_prev + (1.0 - p_s) * (1.0 - b_prev)
        else:
            numerator = (1.0 - p_s) * b_prev
            denominator = (1.0 - p_s) * b_prev + p_s * (1.0 - b_prev)
        b_new = numerator / denominator.clamp_min(self.epsilon)
        self.belief_maps[proc_idx, agent_idx][mask] = b_new.clamp(0.0, 1.0)

    def _detections_tensor(self, local_detections, proc_idx, agent_idx):
        detections = local_detections[proc_idx][agent_idx]
        if isinstance(detections, torch.Tensor):
            return detections.to(device=self.device, dtype=torch.float32).view(-1, 2)
        if len(detections) == 0:
            return torch.zeros(0, 2, device=self.device)
        return torch.as_tensor(
            detections, dtype=torch.float32, device=self.device
        ).view(-1, 2)

    def _step_values(self, step_id, num_processes):
        if isinstance(step_id, torch.Tensor):
            values = step_id.to(device=self.device, dtype=torch.float32).view(-1)
        elif isinstance(step_id, (list, tuple)):
            values = torch.tensor(step_id, device=self.device, dtype=torch.float32)
        else:
            values = torch.full(
                (num_processes,),
                float(step_id),
                device=self.device,
                dtype=torch.float32,
            )
        if values.numel() == 1:
            values = values.repeat(num_processes)
        return values

    def _receiver_first_sources(self, receiver_idx, neighbor_indices):
        receiver = torch.tensor([receiver_idx], device=self.device, dtype=torch.long)
        others = neighbor_indices[neighbor_indices != receiver_idx]
        if others.numel() == 0:
            return receiver
        others, _ = torch.sort(others)
        return torch.cat([receiver, others])
