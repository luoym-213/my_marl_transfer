"""Constant-velocity Kalman teammate estimator."""

from __future__ import division

import itertools

import numpy as np

from estimators.base import (
    EstimateBatch,
    StateEstimator,
    validate_delivery,
    validate_initial_states,
)


def _positive_definite(matrix, epsilon=1e-9):
    matrix = 0.5 * (matrix + matrix.T)
    minimum = float(np.linalg.eigvalsh(matrix).min())
    if minimum < epsilon:
        matrix = matrix + np.eye(matrix.shape[0]) * (epsilon - minimum)
    return matrix


class ConstantVelocityKalmanEstimator(StateEstimator):
    """Per-directed-link CV Kalman filter with non-persistent queries.

    Persistent filters are anchored at their newest accepted *source* time.
    Thus a delayed packet is applied at the state time at which it was
    generated, while an estimate requested for the current arrival time is a
    temporary propagation and cannot feed back into later updates.
    """

    def __init__(
            self, num_agents, process_covariance, observation_covariance,
            dt=0.1, initial_covariance=None):
        super(ConstantVelocityKalmanEstimator, self).__init__(
            num_agents, dt=dt
        )
        self.process_covariance = _positive_definite(
            np.asarray(process_covariance, dtype=np.float64)
        )
        self.observation_covariance = _positive_definite(
            np.asarray(observation_covariance, dtype=np.float64)
        )
        if self.process_covariance.shape != (4, 4):
            raise ValueError("process_covariance must have shape (4, 4)")
        if self.observation_covariance.shape != (2, 2):
            raise ValueError("observation_covariance must have shape (2, 2)")
        if initial_covariance is None:
            initial_covariance = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
        self.initial_covariance = _positive_definite(
            np.asarray(initial_covariance, dtype=np.float64)
        )
        if self.initial_covariance.shape != (4, 4):
            raise ValueError("initial_covariance must have shape (4, 4)")
        self._observation_matrix = np.zeros((2, 4), dtype=np.float64)
        self._observation_matrix[:, :2] = np.eye(2)

    def _transition(self, step_delta):
        transition = np.eye(4, dtype=np.float64)
        elapsed = max(float(step_delta), 0.0) * self.dt
        transition[0, 2] = elapsed
        transition[1, 3] = elapsed
        return transition

    def _propagate(self, mean, covariance, step_delta):
        if step_delta <= 0:
            return mean.copy(), covariance.copy()
        transition = self._transition(step_delta)
        # The base Q is the covariance of a one-simulator-step CV residual.
        process = self.process_covariance * float(step_delta)
        predicted_mean = np.dot(transition, mean)
        predicted_covariance = (
            np.dot(np.dot(transition, covariance), transition.T) + process
        )
        return predicted_mean, _positive_definite(predicted_covariance)

    def reset(self, initial_states, initial_semantics=None, source_step=0):
        states = validate_initial_states(initial_states, self.num_agents)
        self._means = np.repeat(
            states[None, :, :], self.num_agents, axis=0
        ).astype(np.float64)
        self._covariances = np.tile(
            self.initial_covariance,
            (self.num_agents, self.num_agents, 1, 1),
        )
        self._source_steps = np.full(
            (self.num_agents, self.num_agents), int(source_step), dtype=np.int64
        )
        self._is_reset = True

    def _position_update(self, mean, covariance, position):
        observation = self._observation_matrix
        innovation = np.asarray(position, dtype=np.float64) - np.dot(
            observation, mean
        )
        innovation_covariance = (
            np.dot(np.dot(observation, covariance), observation.T)
            + self.observation_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            np.dot(observation, covariance),
        ).T
        updated_mean = mean + np.dot(gain, innovation)

        # Joseph form retains symmetry and positive semidefiniteness under
        # finite precision better than (I-KH)P.
        identity_minus_gain = np.eye(4) - np.dot(gain, observation)
        updated_covariance = (
            np.dot(np.dot(identity_minus_gain, covariance),
                   identity_minus_gain.T)
            + np.dot(
                np.dot(gain, self.observation_covariance), gain.T
            )
        )
        return updated_mean, _positive_definite(updated_covariance)

    def ingest_deliveries(self, deliveries):
        if not getattr(self, "_is_reset", False):
            raise RuntimeError("reset must be called before ingest_deliveries")
        ordered = sorted(
            deliveries,
            key=lambda item: (
                int(item["arrival_step"]),
                int(item.get("sequence", 0)),
            ),
        )
        for event in ordered:
            receiver, sender, source_step, _, state = validate_delivery(
                event, self.num_agents
            )
            anchor_step = int(self._source_steps[receiver, sender])
            if source_step <= anchor_step:
                continue
            mean, covariance = self._propagate(
                self._means[receiver, sender],
                self._covariances[receiver, sender],
                source_step - anchor_step,
            )
            mean, covariance = self._position_update(
                mean, covariance, state[:2]
            )
            self._means[receiver, sender] = mean
            self._covariances[receiver, sender] = covariance
            self._source_steps[receiver, sender] = source_step

    def predict(self, query_step, ego_states, ego_semantics=None):
        if not getattr(self, "_is_reset", False):
            raise RuntimeError("reset must be called before predict")
        query_step = int(query_step)
        ego_states = validate_initial_states(ego_states, self.num_agents)
        means = np.empty_like(self._means)
        covariances = np.empty_like(self._covariances)
        for ego in range(self.num_agents):
            for sender in range(self.num_agents):
                step_delta = max(
                    query_step - int(self._source_steps[ego, sender]), 0
                )
                means[ego, sender], covariances[ego, sender] = self._propagate(
                    self._means[ego, sender],
                    self._covariances[ego, sender],
                    step_delta,
                )
        source_steps = self._source_steps.copy()
        aoi = np.maximum(query_step - source_steps, 0) * self.dt
        indices = np.arange(self.num_agents)
        means[indices, indices] = ego_states
        covariances[indices, indices] = self.initial_covariance
        source_steps[indices, indices] = query_step
        aoi[indices, indices] = 0.0
        return EstimateBatch(means, covariances, aoi, source_steps)

    @classmethod
    def estimate_base_covariances(cls, episodes, dt=0.1):
        """Estimate Q and R only from ideal training trajectories."""
        residuals = []
        positions = []
        transition = np.eye(4, dtype=np.float64)
        transition[0, 2] = dt
        transition[1, 3] = dt
        for episode in episodes:
            states = np.asarray(episode["states"], dtype=np.float64)
            if states.ndim != 3 or states.shape[-1] != 4:
                raise ValueError("episode states must have shape [time, agent, 4]")
            if states.shape[0] > 1:
                predicted = np.einsum("ij,taj->tai", transition, states[:-1])
                residuals.append(states[1:] - predicted)
            positions.append(states[:, :, :2].reshape(-1, 2))
        if not residuals:
            raise ValueError("at least one two-step trajectory is required")
        residual_array = np.concatenate(
            [item.reshape(-1, 4) for item in residuals], axis=0
        )
        position_array = np.concatenate(positions, axis=0)
        process = np.cov(residual_array, rowvar=False, bias=False)
        position_variance = np.var(position_array, axis=0, ddof=1)
        observation = np.diag(np.maximum(position_variance * 1e-4, 1e-9))
        return _positive_definite(process), _positive_definite(observation)

    @classmethod
    def covariance_grid(cls, base_q, base_r, multipliers=(0.25, 1.0, 4.0)):
        for q_multiplier, r_multiplier in itertools.product(
                multipliers, multipliers):
            yield {
                "q_multiplier": float(q_multiplier),
                "r_multiplier": float(r_multiplier),
                "process_covariance": np.asarray(base_q) * q_multiplier,
                "observation_covariance": np.asarray(base_r) * r_multiplier,
            }
