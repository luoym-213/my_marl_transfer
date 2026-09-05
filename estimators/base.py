"""Common runtime contract for Stale, KF, and CommDrop.

All timestamps passed to this module are integer simulator steps.  ``aoi`` in
the returned value is expressed in seconds so it is independent of the
simulator frequency.
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np


class EstimateBatch(object):
    """One ego-specific estimate for every directed teammate relation."""

    def __init__(self, mean, covariance, aoi, source_steps):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.covariance = np.asarray(covariance, dtype=np.float32)
        self.aoi = np.asarray(aoi, dtype=np.float32)
        self.source_steps = np.asarray(source_steps, dtype=np.int64)
        if self.mean.ndim != 3 or self.mean.shape[-1] != 4:
            raise ValueError("mean must have shape [ego, sender, 4]")
        expected_covariance = self.mean.shape[:2] + (4, 4)
        if self.covariance.shape != expected_covariance:
            raise ValueError(
                "covariance must have shape {}, got {}".format(
                    expected_covariance, self.covariance.shape
                )
            )
        if self.aoi.shape != self.mean.shape[:2]:
            raise ValueError("aoi must have shape [ego, sender]")
        if self.source_steps.shape != self.mean.shape[:2]:
            raise ValueError("source_steps must have shape [ego, sender]")

    def as_dict(self):
        return {
            "mean": self.mean.copy(),
            "covariance": self.covariance.copy(),
            "aoi": self.aoi.copy(),
            "source_steps": self.source_steps.copy(),
        }


class StateEstimator(object, metaclass=ABCMeta):
    """The state-estimator interface used by offline and closed-loop code."""

    def __init__(self, num_agents, dt=0.1):
        self.num_agents = int(num_agents)
        self.dt = float(dt)
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @abstractmethod
    def reset(self, initial_states, initial_semantics=None, source_step=0):
        """Reset persistent history from the mandatory t=0 synchronization."""

    @abstractmethod
    def ingest_deliveries(self, deliveries):
        """Consume real delivered packets; implementations reject old packets."""

    @abstractmethod
    def predict(self, query_step, ego_states, ego_semantics=None):
        """Return estimates without modifying persistent estimator history."""


def validate_initial_states(initial_states, num_agents):
    states = np.asarray(initial_states, dtype=np.float32)
    if states.shape == (num_agents, num_agents, 4):
        # A reset info object contains one identical synchronized row per ego.
        states = states[0]
    if states.shape != (num_agents, 4):
        raise ValueError(
            "initial_states must have shape ({}, 4), got {}".format(
                num_agents, states.shape
            )
        )
    return states.copy()


def validate_delivery(event, num_agents):
    required = {
        "receiver", "sender", "source_step", "arrival_step", "motion_state"
    }
    missing = required.difference(event)
    if missing:
        raise ValueError("delivery is missing {}".format(sorted(missing)))
    receiver = int(event["receiver"])
    sender = int(event["sender"])
    if not 0 <= receiver < num_agents or not 0 <= sender < num_agents:
        raise ValueError("delivery sender/receiver is out of range")
    if receiver == sender:
        raise ValueError("communication deliveries may not contain self loops")
    source_step = int(event["source_step"])
    arrival_step = int(event["arrival_step"])
    if arrival_step < source_step:
        raise ValueError("arrival_step must not precede source_step")
    state = np.asarray(event["motion_state"], dtype=np.float32)
    if state.shape != (4,):
        raise ValueError("motion_state must have shape (4,)")
    return receiver, sender, source_step, arrival_step, state


def default_semantics(num_agents, states=None):
    if states is None:
        goals = np.zeros((num_agents, 2), dtype=np.float32)
    else:
        goals = np.asarray(states, dtype=np.float32)[:, :2].copy()
    return {
        "option": np.zeros(num_agents, dtype=np.int64),
        "goal": goals,
        "task_progress": np.zeros(num_agents, dtype=np.float32),
        "active": np.ones(num_agents, dtype=np.float32),
    }


class StaleEstimator(StateEstimator):
    """Hold the newest source-timestamped delivered motion state."""

    def __init__(
            self, num_agents, dt=0.1, initial_variance=1e-4,
            variance_growth=(0.02, 0.02, 0.05, 0.05)):
        super(StaleEstimator, self).__init__(num_agents, dt=dt)
        self.initial_variance = float(initial_variance)
        self.variance_growth = np.asarray(variance_growth, dtype=np.float64)
        if self.variance_growth.shape != (4,):
            raise ValueError("variance_growth must contain four values")

    def reset(self, initial_states, initial_semantics=None, source_step=0):
        states = validate_initial_states(initial_states, self.num_agents)
        self._states = np.repeat(
            states[None, :, :], self.num_agents, axis=0
        ).astype(np.float64)
        self._source_steps = np.full(
            (self.num_agents, self.num_agents), int(source_step), dtype=np.int64
        )
        self._is_reset = True

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
            if source_step <= self._source_steps[receiver, sender]:
                continue
            self._states[receiver, sender] = state
            self._source_steps[receiver, sender] = source_step

    def predict(self, query_step, ego_states, ego_semantics=None):
        if not getattr(self, "_is_reset", False):
            raise RuntimeError("reset must be called before predict")
        query_step = int(query_step)
        ego_states = validate_initial_states(ego_states, self.num_agents)
        means = self._states.copy()
        source_steps = self._source_steps.copy()
        aoi = np.maximum(query_step - source_steps, 0).astype(np.float64) * self.dt
        variances = (
            self.initial_variance
            + aoi[:, :, None] * self.variance_growth[None, None, :]
        )
        covariance = np.zeros(
            (self.num_agents, self.num_agents, 4, 4), dtype=np.float64
        )
        diagonal = np.arange(4)
        covariance[:, :, diagonal, diagonal] = variances

        # Every ego always observes its own live motion state exactly.  This is
        # an output-only substitution and is never written to message history.
        indices = np.arange(self.num_agents)
        means[indices, indices] = ego_states
        source_steps[indices, indices] = query_step
        aoi[indices, indices] = 0.0
        covariance[indices, indices] = np.eye(4) * self.initial_variance
        return EstimateBatch(means, covariance, aoi, source_steps)
