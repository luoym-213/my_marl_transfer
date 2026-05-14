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
        work_device = torch.device("cpu")
        updated_data = prev_landmark_data.detach().to(work_device).clone()
        updated_mask = prev_landmark_mask.detach().to(work_device).clone()
        updated_timestamp = prev_landmark_timestamp.detach().to(work_device).clone()

        num_agents_processes, max_landmarks, _ = updated_data.shape
        num_processes = len(local_detection_list)
        num_agents = num_agents_processes // num_processes
        step_values = self._step_values(step_id, num_processes, work_device)
        comm_mask_cpu = comm_mask.detach().to(work_device).bool()

        if env_dones is not None and env_dones.any():
            proc_indices = torch.arange(num_processes, device=work_device)[
                env_dones.to(work_device)
            ]
            agent_indices = torch.arange(num_agents, device=work_device)
            proc_mesh, agent_mesh = torch.meshgrid(
                proc_indices, agent_indices, indexing="ij"
            )
            linear_indices = agent_mesh.flatten() * num_processes + proc_mesh.flatten()
            updated_mask[linear_indices, :, 0] = 0.0
            updated_data[linear_indices, :, :] = 0.0
            updated_timestamp[linear_indices, :, 0] = -1.0

        self._batch_upsert_local_detections(
            updated_data,
            updated_mask,
            updated_timestamp,
            local_detection_list,
            step_values,
            num_processes,
            num_agents,
            work_device,
            match_threshold,
        )

        self._fuse_by_comm_mask(
            updated_data,
            updated_mask,
            updated_timestamp,
            comm_mask_cpu,
            num_processes,
            num_agents,
            match_threshold,
        )

        return (
            updated_data.to(target_device),
            updated_mask.to(target_device),
            updated_timestamp.to(target_device),
        )

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

    def _batch_upsert_local_detections(
        self,
        data,
        mask,
        timestamp,
        local_detection_list,
        step_values,
        num_processes,
        num_agents,
        device,
        match_threshold,
    ):
        detections, detection_mask = self._detections_batch_tensor(
            local_detection_list,
            num_processes,
            num_agents,
            device,
        )
        if detections.numel() == 0:
            return

        num_rows, max_detections = detections.shape[:2]
        row_indices = torch.arange(num_rows, device=device)
        row_step_values = step_values.repeat(num_agents)

        for det_idx in range(max_detections):
            active = detection_mask[:, det_idx]
            if not active.any():
                continue

            det_pos = detections[:, det_idx]
            valid_slots = mask[:, :, 0] > 0.5
            dists = torch.norm(data[:, :, 0:2] - det_pos.unsqueeze(1), dim=2)
            dists = dists.masked_fill(~valid_slots, float("inf"))
            min_dists, matched_slots = dists.min(dim=1)
            matched = active & (min_dists < match_threshold)
            if matched.any():
                matched_rows = row_indices[matched]
                matched_cols = matched_slots[matched]
                data[matched_rows, matched_cols, 0:2] = det_pos[matched_rows]
                data[matched_rows, matched_cols, 2] = 2.0
                timestamp[matched_rows, matched_cols, 0] = row_step_values[
                    matched_rows
                ]
                mask[matched_rows, matched_cols, 0] = 1.0

            unmatched = active & (~matched)
            if not unmatched.any():
                continue

            empty_mask = mask[:, :, 0] < 0.5
            has_empty = empty_mask.any(dim=1)
            first_empty = empty_mask.float().argmax(dim=1)
            oldest = torch.argmin(timestamp[:, :, 0], dim=1)
            insert_slots = torch.where(has_empty, first_empty, oldest)
            insert_rows = row_indices[unmatched]
            insert_cols = insert_slots[unmatched]
            data[insert_rows, insert_cols, 0:2] = det_pos[insert_rows]
            data[insert_rows, insert_cols, 2] = 2.0
            data[insert_rows, insert_cols, 3] = 0.0
            mask[insert_rows, insert_cols, 0] = 1.0
            timestamp[insert_rows, insert_cols, 0] = row_step_values[
                insert_rows
            ]

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
        max_landmarks = base_data.size(1)
        threshold_sq = match_threshold * match_threshold

        for proc_idx in range(num_processes):
            for receiver_idx in range(num_agents):
                receiver_linear = receiver_idx * num_processes + proc_idx
                candidates = []
                for source_idx in range(num_agents):
                    if not bool(comm_mask[proc_idx, receiver_idx, source_idx]):
                        continue
                    source_linear = source_idx * num_processes + proc_idx
                    for slot_idx in range(max_landmarks):
                        if float(base_mask[source_linear, slot_idx, 0]) <= 0.5:
                            continue
                        candidate_ts = float(
                            base_timestamp[source_linear, slot_idx, 0]
                        )
                        if candidate_ts < 0:
                            continue
                        candidates.append(
                            (
                                -candidate_ts,
                                source_idx,
                                slot_idx,
                                base_data[source_linear, slot_idx],
                                base_timestamp[source_linear, slot_idx, 0],
                                candidate_ts,
                            )
                        )
                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                self._merge_candidates_batch(
                    data[receiver_linear],
                    mask[receiver_linear],
                    timestamp[receiver_linear],
                    candidates,
                    threshold_sq,
                )

    def _merge_candidates_batch(
        self,
        receiver_data,
        receiver_mask,
        receiver_timestamp,
        candidates,
        threshold_sq,
    ):
        max_landmarks = receiver_data.size(0)
        cluster_data = []
        cluster_timestamp = []
        cluster_ts = []
        cluster_order = []

        for slot_idx in range(max_landmarks):
            if float(receiver_mask[slot_idx, 0]) <= 0.5:
                continue
            slot_ts = float(receiver_timestamp[slot_idx, 0])
            if slot_ts < 0:
                continue
            cluster_data.append(receiver_data[slot_idx])
            cluster_timestamp.append(receiver_timestamp[slot_idx, 0])
            cluster_ts.append(slot_ts)
            cluster_order.append(slot_idx)

        next_order = max_landmarks
        for _, source_idx, slot_idx, cand_data, cand_timestamp, candidate_ts in candidates:
            candidate_x = float(cand_data[0])
            candidate_y = float(cand_data[1])
            matched_cluster = None
            best_dist_sq = float("inf")
            for cluster_idx, data_ref in enumerate(cluster_data):
                dx = float(data_ref[0]) - candidate_x
                dy = float(data_ref[1]) - candidate_y
                dist_sq = dx * dx + dy * dy
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    matched_cluster = cluster_idx

            if matched_cluster is not None and best_dist_sq < threshold_sq:
                if candidate_ts > cluster_ts[matched_cluster]:
                    cluster_data[matched_cluster] = cand_data
                    cluster_timestamp[matched_cluster] = cand_timestamp
                    cluster_ts[matched_cluster] = candidate_ts
                continue

            cluster_data.append(cand_data)
            cluster_timestamp.append(cand_timestamp)
            cluster_ts.append(candidate_ts)
            cluster_order.append(next_order + source_idx * max_landmarks + slot_idx)

        if len(cluster_data) > max_landmarks:
            keep_indices = sorted(
                range(len(cluster_data)),
                key=lambda idx: (-cluster_ts[idx], cluster_order[idx]),
            )[:max_landmarks]
        else:
            keep_indices = list(range(len(cluster_data)))

        keep_indices.sort(key=lambda idx: cluster_order[idx])
        if keep_indices:
            output_data = torch.stack([cluster_data[idx] for idx in keep_indices])
            output_timestamp = torch.stack([
                cluster_timestamp[idx] for idx in keep_indices
            ])
        else:
            output_data = None
            output_timestamp = None

        receiver_data.zero_()
        receiver_mask.zero_()
        receiver_timestamp.fill_(-1.0)
        if output_data is None:
            return
        slot_count = output_data.size(0)
        receiver_data[:slot_count] = output_data
        receiver_timestamp[:slot_count, 0] = output_timestamp
        for slot_idx in range(slot_count):
            receiver_mask[slot_idx, 0] = 1.0

    def _merge_candidate(
        self,
        receiver_data,
        receiver_mask,
        receiver_timestamp,
        candidate_data,
        candidate_timestamp,
        match_threshold,
    ):
        candidate_x = float(candidate_data[0])
        candidate_y = float(candidate_data[1])
        candidate_ts = float(candidate_timestamp)
        matched_idx = self._find_landmark_match_scalar(
            candidate_x,
            candidate_y,
            receiver_data,
            receiver_mask,
            match_threshold * match_threshold,
        )

        if matched_idx is not None:
            if candidate_ts > float(receiver_timestamp[matched_idx, 0]):
                receiver_data[matched_idx] = candidate_data
                receiver_mask[matched_idx, 0] = 1.0
                receiver_timestamp[matched_idx, 0] = candidate_timestamp
            return

        empty_idx = self._find_empty_slot_scalar(receiver_mask)
        if empty_idx is None:
            oldest_idx = self._find_oldest_slot_scalar(receiver_timestamp)
            if candidate_ts <= float(receiver_timestamp[oldest_idx, 0]):
                return
            empty_idx = oldest_idx

        receiver_data[empty_idx] = candidate_data
        receiver_mask[empty_idx, 0] = 1.0
        receiver_timestamp[empty_idx, 0] = candidate_timestamp

    def _find_landmark_match_scalar(
        self,
        candidate_x,
        candidate_y,
        landmarks_data,
        landmarks_mask,
        threshold_sq,
    ):
        best_idx = None
        best_dist_sq = float("inf")
        for slot_idx in range(landmarks_data.size(0)):
            if float(landmarks_mask[slot_idx, 0]) <= 0.5:
                continue
            dx = float(landmarks_data[slot_idx, 0]) - candidate_x
            dy = float(landmarks_data[slot_idx, 1]) - candidate_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = slot_idx
        if best_idx is not None and best_dist_sq < threshold_sq:
            return best_idx
        return None

    def _find_empty_slot_scalar(self, landmarks_mask):
        for slot_idx in range(landmarks_mask.size(0)):
            if float(landmarks_mask[slot_idx, 0]) < 0.5:
                return slot_idx
        return None

    def _find_oldest_slot_scalar(self, landmarks_timestamp):
        oldest_idx = 0
        oldest_ts = float(landmarks_timestamp[0, 0])
        for slot_idx in range(1, landmarks_timestamp.size(0)):
            slot_ts = float(landmarks_timestamp[slot_idx, 0])
            if slot_ts < oldest_ts:
                oldest_ts = slot_ts
                oldest_idx = slot_idx
        return oldest_idx

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

    def _detections_batch_tensor(
        self,
        local_detection_list,
        num_processes,
        num_agents,
        device,
    ):
        max_detections = 0
        for proc_idx in range(num_processes):
            for agent_idx in range(num_agents):
                max_detections = max(
                    max_detections,
                    len(local_detection_list[proc_idx][agent_idx]),
                )
        if max_detections == 0:
            return (
                torch.zeros(
                    num_processes * num_agents,
                    0,
                    2,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.zeros(
                    num_processes * num_agents,
                    0,
                    device=device,
                    dtype=torch.bool,
                ),
            )

        positions = torch.zeros(
            num_processes * num_agents,
            max_detections,
            2,
            device=device,
            dtype=torch.float32,
        )
        detection_mask = torch.zeros(
            num_processes * num_agents,
            max_detections,
            device=device,
            dtype=torch.bool,
        )
        for proc_idx in range(num_processes):
            for agent_idx in range(num_agents):
                detections = self._detections_tensor(
                    local_detection_list,
                    proc_idx,
                    agent_idx,
                    device,
                )
                if detections.numel() == 0:
                    continue
                linear_idx = agent_idx * num_processes + proc_idx
                count = detections.size(0)
                positions[linear_idx, :count] = detections
                detection_mask[linear_idx, :count] = True
        return positions, detection_mask

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
