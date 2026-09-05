"""Shared offline/closed-loop estimator metrics and confidence intervals."""

from __future__ import division

import math

import numpy as np


def _off_diagonal(num_agents):
    return ~np.eye(num_agents, dtype=bool)


def _spatial_features(relative):
    """Task-graph spatial edge [relative_x, relative_y, distance]."""
    relative = np.asarray(relative, dtype=np.float64)
    distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    return np.concatenate([relative, distance], axis=-1)


def gaussian_nll(target, mean, covariance, epsilon=1e-8):
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    difference = target - mean
    values = []
    for diff, cov in zip(
            difference.reshape(-1, 4), covariance.reshape(-1, 4, 4)):
        cov = 0.5 * (cov + cov.T) + np.eye(4) * epsilon
        sign, log_determinant = np.linalg.slogdet(cov)
        if sign <= 0:
            values.append(float("inf"))
            continue
        mahalanobis = float(np.dot(diff, np.linalg.solve(cov, diff)))
        values.append(0.5 * (
            4.0 * math.log(2.0 * math.pi) + log_determinant + mahalanobis
        ))
    return np.asarray(values, dtype=np.float64).reshape(difference.shape[:-1])


class EpisodeMetricAccumulator(object):
    def __init__(self, num_agents):
        self.num_agents = int(num_agents)
        self.position_squared = []
        self.velocity_squared = []
        self.edge_squared = []
        self.nll = []

    def add(self, estimate, truth_states, ego_states=None):
        truth = np.asarray(truth_states, dtype=np.float64)
        if ego_states is None:
            ego_states = truth
        ego_states = np.asarray(ego_states, dtype=np.float64)
        target = np.repeat(truth[None, :, :], self.num_agents, axis=0)
        mask = _off_diagonal(self.num_agents)
        difference = np.asarray(estimate.mean, dtype=np.float64) - target
        self.position_squared.extend(
            np.sum(difference[:, :, :2] ** 2, axis=-1)[mask].tolist()
        )
        self.velocity_squared.extend(
            np.sum(difference[:, :, 2:] ** 2, axis=-1)[mask].tolist()
        )
        predicted_edges = (
            np.asarray(estimate.mean[:, :, :2], dtype=np.float64)
            - ego_states[:, None, :2]
        )
        true_edges = truth[None, :, :2] - ego_states[:, None, :2]
        predicted_features = _spatial_features(predicted_edges)
        true_features = _spatial_features(true_edges)
        self.edge_squared.extend(
            np.sum(
                (predicted_features - true_features) ** 2, axis=-1
            )[mask].tolist()
        )
        nll = gaussian_nll(target, estimate.mean, estimate.covariance)
        self.nll.extend(nll[mask].tolist())

    def result(self):
        def root_mean(values):
            return float(np.sqrt(np.mean(values))) if values else float("nan")
        return {
            "position_rmse": root_mean(self.position_squared),
            "velocity_rmse": root_mean(self.velocity_squared),
            "spatial_edge_rmse": root_mean(self.edge_squared),
            "nll": float(np.mean(self.nll)) if self.nll else float("nan"),
        }


def summarize_episode_metrics(episode_metrics):
    if not episode_metrics:
        raise ValueError("at least one episode metric is required")
    summary = {}
    for name in episode_metrics[0]:
        values = np.asarray([item[name] for item in episode_metrics], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[name] = {
                "mean": None, "std": None, "ci95_low": None,
                "ci95_high": None, "episodes": int(values.size),
            }
            continue
        mean = float(finite.mean())
        std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(float(finite.size))
        summary[name] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "episodes": int(values.size),
        }
    return summary


def evaluate_estimator_episodes(estimator_factory, episodes, level, event_builder):
    all_metrics = []
    estimator = estimator_factory()
    for episode in episodes:
        states = np.asarray(episode["states"], dtype=np.float32)
        num_agents = states.shape[1]
        initial_semantics = {
            "option": episode["options"][0],
            "goal": episode["goals"][0],
            "task_progress": episode["task_progress"][0],
            "active": episode["active"][0],
        }
        estimator.reset(states[0], initial_semantics, source_step=0)
        deliveries = event_builder(episode, level)
        accumulator = EpisodeMetricAccumulator(num_agents)
        for query_step in range(states.shape[0]):
            estimator.ingest_deliveries(deliveries[query_step])
            ego_semantics = {
                "option": episode["options"][query_step],
                "goal": episode["goals"][query_step],
                "task_progress": episode["task_progress"][query_step],
                "active": episode["active"][query_step],
            }
            estimate = estimator.predict(
                query_step, states[query_step], ego_semantics
            )
            accumulator.add(estimate, states[query_step], states[query_step])
        all_metrics.append(accumulator.result())
    return all_metrics
