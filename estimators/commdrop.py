"""Paper-faithful CommDrop equations (3.9)--(3.35).

Persistent histories are updated only by real, source-newer packet arrivals.
Queries, graph correction, and predictions are pure and never feed estimates
back into communication history.
"""

from __future__ import division

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from estimators.base import (
    EstimateBatch,
    StateEstimator,
    default_semantics,
    validate_delivery,
    validate_initial_states,
)


DEFAULT_COMMDROP_CONFIG = {
    "hidden_dim": 128,
    "task_dim": 64,
    "time_dim": 16,
    "min_omega": 0.1,
    "max_omega": 5.0,
    "delta_source_seconds": 0.30,
    "lambda_q": 1.0,
    "lambda_edge_task": 1.0,
    "kappa": 3.0,
    "lambda_sparse": 1e-3,
    "epsilon": 1e-4,
    "num_options": 2,
    "dt": 0.1,
}


def _merged_config(config=None):
    result = dict(DEFAULT_COMMDROP_CONFIG)
    if config:
        result.update(config)
    time_dim = int(result["time_dim"])
    if time_dim <= 0 or time_dim % 2:
        raise ValueError("time_dim must be a positive even number")
    return result


class FourierTimeEncoding(nn.Module):
    """Equations (3.9)--(3.10): [sin(delta/omega), cos(delta/omega)]."""

    def __init__(self, dimension=16, min_omega=0.1, max_omega=5.0):
        super(FourierTimeEncoding, self).__init__()
        dimension = int(dimension)
        if dimension <= 0 or dimension % 2:
            raise ValueError("Fourier time dimension must be positive and even")
        omega = torch.logspace(
            math.log10(float(min_omega)),
            math.log10(float(max_omega)),
            steps=dimension // 2,
        )
        self.register_buffer("omega", omega)
        self.dimension = dimension

    def forward(self, seconds):
        seconds = torch.as_tensor(
            seconds, dtype=self.omega.dtype, device=self.omega.device
        )
        phase = seconds.unsqueeze(-1) / self.omega
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class CommDropModel(nn.Module):
    """Event prior, dynamic dependency graph, and Gaussian decoder."""

    def __init__(self, normalization, config=None):
        super(CommDropModel, self).__init__()
        self.config = _merged_config(config)
        hidden = int(self.config["hidden_dim"])
        task_dim = int(self.config["task_dim"])
        time_dim = int(self.config["time_dim"])
        option_count = int(self.config["num_options"])
        self.hidden_dim = hidden
        self.task_dim = task_dim
        self.time_dim = time_dim

        self._register_statistics(normalization)
        self.time_encoding = FourierTimeEncoding(
            time_dim, self.config["min_omega"], self.config["max_omega"]
        )

        # (3.11)--(3.12): absolute sender state, same-source relative state,
        # and the interval between consecutive accepted source timestamps.
        self.event_mlp = nn.Sequential(
            nn.Linear(8 + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.history_gru = nn.GRUCell(hidden, hidden)
        # (3.13): persistent history, latest event, and current AoI.
        self.aoi_query = nn.Sequential(
            nn.Linear(2 * hidden + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.self_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        # (3.18): option and activity are embedded before continuous task
        # goal coordinates and normalized goal distance/progress are appended.
        embedding_dim = max(4, min(16, task_dim // 4))
        self.option_embedding = nn.Embedding(option_count, embedding_dim)
        self.active_embedding = nn.Embedding(2, embedding_dim)
        self.task_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 3, task_dim),
            nn.ReLU(),
            nn.Linear(task_dim, task_dim),
        )

        # (3.17) and (3.19).
        state_edge_dim = 4 * hidden + 3 * time_dim
        task_edge_dim = 4 * task_dim
        self.state_edge_score = nn.Sequential(
            nn.Linear(state_edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.task_edge_score = nn.Sequential(
            nn.Linear(task_edge_dim, task_dim),
            nn.ReLU(),
            nn.Linear(task_dim, 1),
        )
        # (3.23): relation-specific latent residual, not a state prediction.
        self.message_network = nn.Sequential(
            nn.Linear(state_edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # (3.28)--(3.30).  Only the variance decoder additionally observes
        # AoI and effective incoming edge weight Z.
        self.mean_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4)
        )
        self.variance_decoder = nn.Sequential(
            nn.Linear(hidden + time_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def _register_statistics(self, normalization):
        state = normalization["state"]
        goal = normalization["goal"]
        progress = normalization["task_progress"]
        self.register_buffer(
            "state_mean", torch.as_tensor(state["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "state_std", torch.as_tensor(state["std"], dtype=torch.float32)
        )
        self.register_buffer(
            "goal_mean", torch.as_tensor(goal["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "goal_std", torch.as_tensor(goal["std"], dtype=torch.float32)
        )
        self.register_buffer(
            "progress_mean",
            torch.as_tensor(progress["mean"], dtype=torch.float32).reshape(1),
        )
        self.register_buffer(
            "progress_std",
            torch.as_tensor(progress["std"], dtype=torch.float32).reshape(1),
        )

    def normalization_dict(self):
        return {
            "state": {
                "mean": self.state_mean.detach().cpu().numpy().copy(),
                "std": self.state_std.detach().cpu().numpy().copy(),
            },
            "goal": {
                "mean": self.goal_mean.detach().cpu().numpy().copy(),
                "std": self.goal_std.detach().cpu().numpy().copy(),
            },
            "task_progress": {
                "mean": self.progress_mean.detach().cpu().numpy().copy(),
                "std": self.progress_std.detach().cpu().numpy().copy(),
            },
        }

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state):
        return state * self.state_std + self.state_mean

    def event_features(self, sender_state, ego_source_state, interval_seconds):
        sender_state = torch.as_tensor(
            sender_state, dtype=self.state_mean.dtype, device=self.state_mean.device
        )
        ego_source_state = torch.as_tensor(
            ego_source_state,
            dtype=self.state_mean.dtype,
            device=self.state_mean.device,
        )
        relative_state = (sender_state - ego_source_state) / self.state_std
        return torch.cat([
            self.normalize_state(sender_state),
            relative_state,
            self.time_encoding(interval_seconds),
        ], dim=-1)

    def encode_event(self, sender_state, ego_source_state, interval_seconds):
        return self.event_mlp(self.event_features(
            sender_state, ego_source_state, interval_seconds
        ))

    def update_history(self, previous_hidden, encoded_event):
        return self.history_gru(encoded_event, previous_hidden)

    def task_features(self, option, goal, progress, active):
        device = self.state_mean.device
        option = torch.as_tensor(option, device=device).long().clamp(
            0, int(self.config["num_options"]) - 1
        )
        active = torch.as_tensor(active, device=device)
        active_index = (active > 0.5).long()
        goal = torch.as_tensor(
            goal, dtype=self.state_mean.dtype, device=device
        )
        progress = torch.as_tensor(
            progress, dtype=self.state_mean.dtype, device=device
        ).unsqueeze(-1)
        return torch.cat([
            self.option_embedding(option),
            self.active_embedding(active_index),
            (goal - self.goal_mean) / self.goal_std,
            (progress - self.progress_mean) / self.progress_std,
        ], dim=-1)

    @staticmethod
    def _pair_features(node_features):
        count = node_features.shape[1]
        source = node_features.unsqueeze(2).expand(-1, -1, count, -1)
        target = node_features.unsqueeze(1).expand(-1, count, -1, -1)
        return source, target, torch.cat([
            source, target, source - target, source * target
        ], dim=-1)

    def predict_from_memory(
            self, hidden, latest_events, options, goals, progress, active,
            source_steps, initialized, query_step, ego_states):
        """Pure equations (3.13)--(3.32) query on [ego, UAV, ...] tensors."""
        device = self.state_mean.device
        dtype = self.state_mean.dtype
        hidden = hidden.to(device=device, dtype=dtype)
        latest_events = latest_events.to(device=device, dtype=dtype)
        source_steps = source_steps.to(device=device, dtype=dtype)
        initialized = initialized.to(device=device, dtype=torch.bool)
        ego_states = ego_states.to(device=device, dtype=dtype)
        num_egos, num_agents = source_steps.shape
        if num_egos != num_agents:
            raise ValueError("CommDrop expects one ego memory row per UAV")
        dt = float(self.config["dt"])
        aoi = torch.clamp(float(query_step) - source_steps, min=0.0) * dt
        indices = torch.arange(num_agents, device=device)
        # (3.14): ego is always live even though its packet history is not.
        aoi = aoi.clone()
        aoi[indices, indices] = 0.0
        aoi_code = self.time_encoding(aoi)
        priors = self.aoi_query(torch.cat([
            hidden, latest_events, aoi_code
        ], dim=-1))
        priors = priors.clone()
        priors[indices, indices] = self.self_encoder(
            self.normalize_state(ego_states)
        )

        task = self.task_encoder(self.task_features(
            options, goals, progress, active
        ))
        _, _, state_pair = self._pair_features(priors)
        source_aoi = aoi.unsqueeze(2).expand(-1, -1, num_agents)
        target_aoi = aoi.unsqueeze(1).expand(-1, num_agents, -1)
        state_time_edge = torch.cat([
            state_pair,
            self.time_encoding(source_aoi),
            self.time_encoding(target_aoi),
            self.time_encoding(torch.abs(source_aoi - target_aoi)),
        ], dim=-1)
        _, _, task_edge = self._pair_features(task)

        # (3.20)--(3.21): task semantics decay with both endpoint AoIs.
        beta = torch.exp(
            -float(self.config["lambda_q"]) * (source_aoi + target_aoi)
        )
        gate_logits = (
            self.state_edge_score(state_time_edge).squeeze(-1)
            + float(self.config["lambda_edge_task"])
            * beta
            * self.task_edge_score(task_edge).squeeze(-1)
        )
        no_self_loop = ~torch.eye(
            num_agents, dtype=torch.bool, device=device
        ).unsqueeze(0)
        candidate_edges = (
            no_self_loop
            & initialized.unsqueeze(2)
            & initialized.unsqueeze(1)
        )
        edge_gates = torch.sigmoid(gate_logits) * candidate_edges.to(dtype)

        # (3.22): ego plus only teammates with sufficiently fresh AoI may
        # serve as message sources.  Stale nodes remain valid targets.
        source_ids = torch.arange(num_agents, device=device)[None, :]
        ego_ids = torch.arange(num_egos, device=device)[:, None]
        fresh_source = (
            (source_ids == ego_ids)
            | (aoi <= float(self.config["delta_source_seconds"]) + 1e-7)
        )
        aggregation_edges = (
            candidate_edges
            & fresh_source.unsqueeze(2)
        )
        effective_gates = edge_gates * aggregation_edges.to(dtype)
        messages = (
            effective_gates.unsqueeze(-1)
            * self.message_network(state_time_edge)
        )
        # (3.24)--(3.25): do not amplify a weak total gate mass below one.
        incoming_weight = effective_gates.sum(dim=1)
        denominator = torch.clamp(incoming_weight, min=1.0).unsqueeze(-1)
        residual = messages.sum(dim=1) / denominator

        # (3.26)--(3.27): correction increases deterministically with target AoI.
        aoi_gate = 1.0 - torch.exp(
            -float(self.config["kappa"]) * aoi
        )
        corrected = priors + aoi_gate.unsqueeze(-1) * residual

        normalized_mean = self.mean_decoder(corrected)
        normalized_variance = F.softplus(self.variance_decoder(torch.cat([
            corrected, aoi_code, incoming_weight.unsqueeze(-1)
        ], dim=-1))) + float(self.config["epsilon"])
        mean = self.denormalize_state(normalized_mean)
        variance = normalized_variance * self.state_std.square()
        return {
            "mean": mean,
            "variance": variance,
            "normalized_mean": normalized_mean,
            "normalized_variance": normalized_variance,
            "aoi": aoi,
            "edge_gates": edge_gates,
            "edge_mask": aggregation_edges,
            "candidate_edge_mask": candidate_edges,
            "incoming_weight": incoming_weight,
            "normalization_denominator": denominator.squeeze(-1),
            "aoi_residual_gate": aoi_gate,
        }


def _semantics_arrays(semantics, num_agents, states):
    values = default_semantics(num_agents, states)
    if semantics:
        values.update(semantics)
    option = np.asarray(values["option"], dtype=np.int64)
    goal = np.asarray(values["goal"], dtype=np.float32)
    progress = np.asarray(values["task_progress"], dtype=np.float32)
    active = np.asarray(values["active"], dtype=np.float32)
    if option.shape == (num_agents, num_agents):
        option = option[0]
    if goal.shape == (num_agents, num_agents, 2):
        goal = goal[0]
    if progress.shape == (num_agents, num_agents):
        progress = progress[0]
    if active.shape == (num_agents, num_agents):
        active = active[0]
    if option.shape != (num_agents,) or goal.shape != (num_agents, 2):
        raise ValueError("invalid semantics shapes")
    if progress.shape != (num_agents,) or active.shape != (num_agents,):
        raise ValueError("invalid semantics shapes")
    return option, goal, progress, active


class CommDropEstimator(StateEstimator):
    """Online stateful wrapper; model queries remain non-persistent."""

    def __init__(self, num_agents, model, device="cpu"):
        super(CommDropEstimator, self).__init__(
            num_agents, dt=float(model.config["dt"])
        )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.last_edge_gates = None

    def reset(self, initial_states, initial_semantics=None, source_step=0):
        states = validate_initial_states(initial_states, self.num_agents)
        option, goal, progress, active = _semantics_arrays(
            initial_semantics, self.num_agents, states
        )
        repeated_states = np.repeat(states[None, :, :], self.num_agents, axis=0)
        self._last_states = torch.as_tensor(
            repeated_states, dtype=torch.float32, device=self.device
        )
        self._options = torch.as_tensor(
            np.repeat(option[None, :], self.num_agents, axis=0),
            dtype=torch.long, device=self.device,
        )
        self._goals = torch.as_tensor(
            np.repeat(goal[None, :, :], self.num_agents, axis=0),
            dtype=torch.float32, device=self.device,
        )
        self._progress = torch.as_tensor(
            np.repeat(progress[None, :], self.num_agents, axis=0),
            dtype=torch.float32, device=self.device,
        )
        self._active = torch.as_tensor(
            np.repeat(active[None, :], self.num_agents, axis=0),
            dtype=torch.float32, device=self.device,
        )
        self._source_steps = torch.full(
            (self.num_agents, self.num_agents), float(source_step),
            dtype=torch.float32, device=self.device,
        )
        self._initialized = torch.ones(
            self.num_agents, self.num_agents,
            dtype=torch.bool, device=self.device,
        )
        sender = self._last_states.reshape(-1, 4)
        ego_source = torch.as_tensor(
            np.repeat(states[:, None, :], self.num_agents, axis=1).reshape(-1, 4),
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            event = self.model.encode_event(
                sender,
                ego_source,
                torch.zeros(self.num_agents * self.num_agents, device=self.device),
            )
            hidden = self.model.update_history(
                torch.zeros_like(event), event
            )
        self._latest_events = event.reshape(
            self.num_agents, self.num_agents, self.model.hidden_dim
        )
        self._hidden = hidden.reshape(
            self.num_agents, self.num_agents, self.model.hidden_dim
        )
        self._is_reset = True

    def ingest_deliveries(self, deliveries):
        if not getattr(self, "_is_reset", False):
            raise RuntimeError("reset must be called before ingest_deliveries")
        ordered = sorted(deliveries, key=lambda event: (
            int(event["arrival_step"]), int(event.get("sequence", 0))
        ))
        with torch.no_grad():
            for event in ordered:
                ego, sender, source_step, _, state = validate_delivery(
                    event, self.num_agents
                )
                previous_step = float(self._source_steps[ego, sender])
                if source_step <= previous_step:
                    continue
                if "ego_source_motion_state" not in event:
                    raise ValueError(
                        "CommDrop delivery requires ego_source_motion_state"
                    )
                interval = (source_step - previous_step) * self.dt
                encoded = self.model.encode_event(
                    torch.as_tensor(
                        state, dtype=torch.float32, device=self.device
                    ).reshape(1, 4),
                    torch.as_tensor(
                        event["ego_source_motion_state"],
                        dtype=torch.float32, device=self.device,
                    ).reshape(1, 4),
                    torch.as_tensor([interval], device=self.device),
                )
                self._hidden[ego, sender] = self.model.update_history(
                    self._hidden[ego, sender].reshape(1, -1), encoded
                )[0]
                self._latest_events[ego, sender] = encoded[0]
                self._last_states[ego, sender] = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                )
                self._options[ego, sender] = int(event.get("option", 0))
                self._goals[ego, sender] = torch.as_tensor(
                    event.get("goal", state[:2]),
                    dtype=torch.float32, device=self.device,
                )
                self._progress[ego, sender] = float(
                    event.get("task_progress", 0.0)
                )
                self._active[ego, sender] = float(event.get("active", 1.0))
                self._source_steps[ego, sender] = float(source_step)

    def memory_snapshot(self):
        return {
            "hidden": self._hidden.detach().clone(),
            "latest_events": self._latest_events.detach().clone(),
            "last_states": self._last_states.detach().clone(),
            "source_steps": self._source_steps.detach().clone(),
        }

    def _query_semantics(self, ego_semantics, ego_states):
        options = self._options.clone()
        goals = self._goals.clone()
        progress = self._progress.clone()
        active = self._active.clone()
        option_now, goal_now, progress_now, active_now = _semantics_arrays(
            ego_semantics, self.num_agents, ego_states
        )
        indices = torch.arange(self.num_agents, device=self.device)
        options[indices, indices] = torch.as_tensor(option_now, device=self.device)
        goals[indices, indices] = torch.as_tensor(goal_now, device=self.device)
        progress[indices, indices] = torch.as_tensor(
            progress_now, device=self.device
        )
        active[indices, indices] = torch.as_tensor(active_now, device=self.device)
        return options, goals, progress, active

    def predict(self, query_step, ego_states, ego_semantics=None):
        if not getattr(self, "_is_reset", False):
            raise RuntimeError("reset must be called before predict")
        ego_states = validate_initial_states(ego_states, self.num_agents)
        options, goals, progress, active = self._query_semantics(
            ego_semantics, ego_states
        )
        with torch.no_grad():
            prediction = self.model.predict_from_memory(
                self._hidden,
                self._latest_events,
                options,
                goals,
                progress,
                active,
                self._source_steps,
                self._initialized,
                int(query_step),
                torch.as_tensor(
                    ego_states, dtype=torch.float32, device=self.device
                ),
            )
        means = prediction["mean"].detach().cpu().numpy()
        variances = prediction["variance"].detach().cpu().numpy()
        aoi = prediction["aoi"].detach().cpu().numpy()
        source_steps = self._source_steps.detach().cpu().numpy().astype(np.int64)
        covariance = np.zeros(
            (self.num_agents, self.num_agents, 4, 4), dtype=np.float32
        )
        diagonal = np.arange(4)
        covariance[:, :, diagonal, diagonal] = variances
        indices = np.arange(self.num_agents)
        means[indices, indices] = ego_states
        covariance[indices, indices] = np.eye(4, dtype=np.float32) * float(
            self.model.config["epsilon"]
        )
        aoi[indices, indices] = 0.0
        source_steps[indices, indices] = int(query_step)
        self.last_edge_gates = prediction["edge_gates"].detach().cpu().numpy()
        return EstimateBatch(means, covariance, aoi, source_steps)

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(
                checkpoint, map_location=lambda storage, location: storage
            )
        model = CommDropModel(
            checkpoint["normalization"], checkpoint["network_config"]
        )
        model.load_state_dict(checkpoint["model_state"])
        return cls(int(checkpoint["num_agents"]), model, device=device)


def commdrop_loss(model, prediction, target_states):
    """Equations (3.33)--(3.35): normalized Gaussian NLL plus sparsity."""
    target_states = target_states.to(
        device=model.state_mean.device, dtype=model.state_mean.dtype
    )
    normalized_target = model.normalize_state(target_states).unsqueeze(0)
    normalized_target = normalized_target.expand_as(prediction["normalized_mean"])
    mean = prediction["normalized_mean"]
    variance = prediction["normalized_variance"].clamp_min(
        float(model.config["epsilon"])
    )
    num_agents = mean.shape[1]
    off_diagonal = ~torch.eye(
        num_agents, dtype=torch.bool, device=mean.device
    )
    per_state = 0.5 * (
        (normalized_target - mean).square() / variance + torch.log(variance)
    ).sum(dim=-1)
    state_loss = per_state[off_diagonal].mean()
    candidate = prediction["candidate_edge_mask"]
    if bool(candidate.any()):
        sparse_loss = prediction["edge_gates"][candidate].mean()
    else:
        sparse_loss = state_loss * 0.0
    total = state_loss + float(model.config["lambda_sparse"]) * sparse_loss
    return {
        "loss": total,
        "nll": state_loss,
        "sparse_loss": sparse_loss,
    }
