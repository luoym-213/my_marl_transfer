import torch


class LandmarkMemory:
    """Updates per-agent landmark memory tensors from detected map points."""

    def __init__(self, device):
        self.device = device

    def update(
        self,
        prev_landmark_data,
        prev_landmark_mask,
        detected_map_list,
        device=None,
        env_dones=None,
        match_threshold=0.1,
        cleanup_threshold=0.06,
    ):
        target_device = self.device if device is None else device
        updated_landmark_data = prev_landmark_data.clone()
        updated_landmark_mask = prev_landmark_mask.clone()

        num_agents_processes, max_landmarks, _ = prev_landmark_data.shape
        num_processes = len(detected_map_list)
        num_agents = num_agents_processes // num_processes

        if env_dones is not None and env_dones.any():
            proc_indices = torch.arange(num_processes, device=self.device)[env_dones]
            agent_indices = torch.arange(num_agents, device=self.device)
            proc_mesh, agent_mesh = torch.meshgrid(
                proc_indices, agent_indices, indexing="ij"
            )
            linear_indices = agent_mesh.flatten() * num_processes + proc_mesh.flatten()
            updated_landmark_mask[linear_indices, :, 0] = 0.0
            updated_landmark_data[linear_indices, :, :] = 0.0

        for proc_idx, detected_map in enumerate(detected_map_list):
            num_detected = detected_map.shape[0]
            linear_idx = proc_idx
            current_landmarks = updated_landmark_data[linear_idx]
            current_mask = updated_landmark_mask[linear_idx]

            if num_detected == 0:
                updated_landmark_mask[linear_idx, :, 0] = 0.0
                updated_landmark_data[linear_idx, :, :] = 0.0
                continue

            for lm_idx in range(max_landmarks):
                if current_mask[lm_idx, 0] < 0.5:
                    continue

                lm_pos = current_landmarks[lm_idx, :2]
                distances_to_detected = torch.norm(
                    detected_map - lm_pos.unsqueeze(0), dim=1
                )
                min_dist = distances_to_detected.min()

                if min_dist > cleanup_threshold:
                    updated_landmark_mask[linear_idx, lm_idx, 0] = 0.0
                    updated_landmark_data[linear_idx, lm_idx, 3] = 0.0

            for det_pos in detected_map:
                matched_idx = self._find_landmark_match(
                    det_pos, current_landmarks, current_mask, match_threshold
                )

                if matched_idx is not None:
                    old_pos = updated_landmark_data[linear_idx, matched_idx, 0:2]
                    updated_landmark_data[linear_idx, matched_idx, 0:2] = (
                        0.3 * old_pos + 0.7 * det_pos
                    )
                else:
                    empty_idx = self._find_empty_slot(updated_landmark_mask[linear_idx])

                    if empty_idx is not None:
                        updated_landmark_data[linear_idx, empty_idx, 0:2] = det_pos
                        updated_landmark_data[linear_idx, empty_idx, 2] = 2.0
                        updated_landmark_data[linear_idx, empty_idx, 3] = 0.0
                        updated_landmark_mask[linear_idx, empty_idx, 0] = 1.0
                    else:
                        print(
                            f"Warning: No empty slot for new landmark at process {proc_idx}"
                        )

            for agent_idx in range(1, num_agents):
                broadcast_linear_idx = agent_idx * num_processes + proc_idx
                updated_landmark_data[broadcast_linear_idx] = updated_landmark_data[
                    linear_idx
                ].clone()
                updated_landmark_mask[broadcast_linear_idx] = updated_landmark_mask[
                    linear_idx
                ].clone()

        return updated_landmark_data.to(target_device), updated_landmark_mask.to(
            target_device
        )

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
