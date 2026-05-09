import torch


class LandmarkMemory:
    """Updates per-agent landmark memory with timestamp-limited sharing.

    Tensors are stored in agent-major rollout order:
        landmark_data: [A*P, M, 4] with [x, y, utility, is_targeted]
        landmark_mask: [A*P, M, 1]
        landmark_timestamp: [A*P, M, 1]
    """

    def __init__(self, device):
        self.device = device

    def update(
        self,
        prev_landmark_data,
        prev_landmark_mask,
        prev_landmark_timestamp,
        local_detection_list,
        comm_mask,
        step_id,
        device=None,
        env_dones=None,
        match_threshold=0.05,
    ):
        target_device = self.device if device is None else device
        updated_data = prev_landmark_data.clone().to(target_device)
        updated_mask = prev_landmark_mask.clone().to(target_device)
        updated_timestamp = prev_landmark_timestamp.clone().to(target_device)

        num_agents_processes, max_landmarks, _ = updated_data.shape
        num_processes = len(local_detection_list)
        num_agents = num_agents_processes // num_processes
        step_values = self._step_values(step_id, num_processes, target_device)

        if env_dones is not None and env_dones.any():
            proc_indices = torch.arange(num_processes, device=target_device)[
                env_dones.to(target_device)
            ]
            agent_indices = torch.arange(num_agents, device=target_device)
            proc_mesh, agent_mesh = torch.meshgrid(
                proc_indices, agent_indices, indexing="ij"
            )
            linear_indices = agent_mesh.flatten() * num_processes + proc_mesh.flatten()
            updated_mask[linear_indices, :, 0] = 0.0
            updated_data[linear_indices, :, :] = 0.0
            updated_timestamp[linear_indices, :, 0] = -1.0

        for proc_idx in range(num_processes):
            for agent_idx in range(num_agents):
                linear_idx = agent_idx * num_processes + proc_idx
                detections = self._detections_tensor(
                    local_detection_list, proc_idx, agent_idx, target_device
                )
                for det_pos in detections:
                    self._upsert_local_detection(
                        updated_data[linear_idx],
                        updated_mask[linear_idx],
                        updated_timestamp[linear_idx],
                        det_pos,
                        step_values[proc_idx],
                        match_threshold,
                    )

        self._fuse_by_comm_mask(
            updated_data,
            updated_mask,
            updated_timestamp,
            comm_mask.to(target_device).bool(),
            num_processes,
            num_agents,
            match_threshold,
        )

        return updated_data, updated_mask, updated_timestamp

    def _upsert_local_detection(
        self,
        data,
        mask,
        timestamp,
        det_pos,
        step_value,
        match_threshold,
    ):
        matched_idx = self._find_landmark_match(det_pos, data, mask, match_threshold)
        if matched_idx is not None:
            data[matched_idx, 0:2] = det_pos
            data[matched_idx, 2] = 2.0
            timestamp[matched_idx, 0] = step_value
            mask[matched_idx, 0] = 1.0
            return

        empty_idx = self._find_empty_slot(mask)
        if empty_idx is None:
            empty_idx = torch.argmin(timestamp[:, 0]).item()

        data[empty_idx, 0:2] = det_pos
        data[empty_idx, 2] = 2.0
        data[empty_idx, 3] = 0.0
        mask[empty_idx, 0] = 1.0
        timestamp[empty_idx, 0] = step_value

    def _fuse_by_comm_mask(
        self,
        data,
        mask,
        timestamp,
        comm_mask,
        num_processes,
        num_agents,
        match_threshold,
    ):
        base_data = data.clone()
        base_mask = mask.clone()
        base_timestamp = timestamp.clone()

        for proc_idx in range(num_processes):
            for receiver_idx in range(num_agents):
                receiver_linear = receiver_idx * num_processes + proc_idx
                source_indices = torch.nonzero(
                    comm_mask[proc_idx, receiver_idx], as_tuple=False
                ).flatten()
                if source_indices.numel() == 0:
                    continue

                candidates = []
                for source_idx in source_indices.tolist():
                    source_linear = source_idx * num_processes + proc_idx
                    for slot_idx in range(base_data.shape[1]):
                        if base_mask[source_linear, slot_idx, 0] < 0.5:
                            continue
                        candidates.append((
                            float(base_timestamp[source_linear, slot_idx, 0].item()),
                            source_idx,
                            slot_idx,
                            base_data[source_linear, slot_idx].clone(),
                        ))

                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                for cand_timestamp, _, _, cand_data in candidates:
                    if cand_timestamp < 0:
                        continue
                    cand_ts = torch.tensor(
                        cand_timestamp, device=data.device, dtype=timestamp.dtype
                    )
                    self._merge_candidate(
                        data[receiver_linear],
                        mask[receiver_linear],
                        timestamp[receiver_linear],
                        cand_data,
                        cand_ts,
                        match_threshold,
                    )

    def _merge_candidate(
        self,
        receiver_data,
        receiver_mask,
        receiver_timestamp,
        candidate_data,
        candidate_timestamp,
        match_threshold,
    ):
        candidate_pos = candidate_data[0:2]
        matched_idx = self._find_landmark_match(
            candidate_pos,
            receiver_data,
            receiver_mask,
            match_threshold,
        )

        if matched_idx is not None:
            if candidate_timestamp > receiver_timestamp[matched_idx, 0]:
                receiver_data[matched_idx] = candidate_data
                receiver_mask[matched_idx, 0] = 1.0
                receiver_timestamp[matched_idx, 0] = candidate_timestamp
            return

        empty_idx = self._find_empty_slot(receiver_mask)
        if empty_idx is None:
            oldest_idx = torch.argmin(receiver_timestamp[:, 0]).item()
            if candidate_timestamp <= receiver_timestamp[oldest_idx, 0]:
                return
            empty_idx = oldest_idx

        receiver_data[empty_idx] = candidate_data
        receiver_mask[empty_idx, 0] = 1.0
        receiver_timestamp[empty_idx, 0] = candidate_timestamp

    def _find_landmark_match(self, position, landmarks_data, landmarks_mask, threshold):
        valid_mask = landmarks_mask[:, 0] > 0.5
        if not valid_mask.any():
            return None

        landmark_positions = landmarks_data[:, 0:2]
        distances = torch.norm(landmark_positions - position.unsqueeze(0), dim=1)
        distances = distances.masked_fill(~valid_mask, float("inf"))
        min_dist, min_idx = distances.min(dim=0)
        if min_dist < threshold:
            return min_idx.item()
        return None

    def _find_empty_slot(self, landmarks_mask):
        empty_mask = landmarks_mask[:, 0] < 0.5
        if empty_mask.any():
            return empty_mask.nonzero(as_tuple=False)[0].item()
        return None

    def _detections_tensor(self, local_detection_list, proc_idx, agent_idx, device):
        detections = local_detection_list[proc_idx][agent_idx]
        if isinstance(detections, torch.Tensor):
            return detections.to(device=device, dtype=torch.float32).view(-1, 2)
        if len(detections) == 0:
            return torch.zeros(0, 2, device=device)
        return torch.as_tensor(detections, dtype=torch.float32, device=device).view(-1, 2)

    def _step_values(self, step_id, num_processes, device):
        if isinstance(step_id, torch.Tensor):
            values = step_id.to(device=device, dtype=torch.float32).view(-1)
        elif isinstance(step_id, (list, tuple)):
            values = torch.tensor(step_id, device=device, dtype=torch.float32)
        else:
            values = torch.full(
                (num_processes,), float(step_id), device=device, dtype=torch.float32
            )
        if values.numel() == 1:
            values = values.repeat(num_processes)
        return values
