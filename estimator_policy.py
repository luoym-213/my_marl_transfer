"""Frozen HIG-SAR policy adapter for teammate state estimators."""

from __future__ import division

import copy
import types

import numpy as np

from communication_baseline import stale_eval_act
from estimators.base import StaleEstimator
from estimators.commdrop import CommDropEstimator
from estimators.training import load_kf_checkpoint


FORBIDDEN_POLICY_INFO_KEYS = {
    "true_agent_states",
    "synchronized_agent_states",
    "ground_truth_agent_states",
}


def _diagonal_rows(values):
    values = np.asarray(values)
    indices = np.arange(values.shape[0])
    return values[indices, indices]


def _current_semantics(info):
    return {
        "option": _diagonal_rows(info["local_agent_tasks"])[..., 0],
        "goal": _diagonal_rows(info["local_agent_goals"]),
        "task_progress": _diagonal_rows(info["local_task_progress"]),
        "active": _diagonal_rows(info["local_agent_active"]),
    }


def _initial_semantics(info):
    return {
        "option": np.asarray(info["local_agent_tasks"])[0, :, 0],
        "goal": np.asarray(info["local_agent_goals"])[0],
        "task_progress": np.asarray(info["local_task_progress"])[0],
        "active": np.asarray(info["local_agent_active"])[0],
    }


def rewrite_low_level_teammates(observations, estimated_states):
    """Replace only the low-level teammate-position observation segment."""
    observations = np.asarray(observations, dtype=np.float32).copy()
    estimated_states = np.asarray(estimated_states, dtype=np.float32)
    num_agents = estimated_states.shape[0]
    base_size = 4 + 2 * num_agents + 2 * (num_agents - 1)
    identity_size = observations.shape[1] - base_size
    if identity_size < 0:
        raise ValueError("unexpected simple_spread observation layout")
    teammate_start = identity_size + 4 + 2 * num_agents
    for ego in range(num_agents):
        offset = teammate_start
        ego_position = estimated_states[ego, ego, :2]
        for sender in range(num_agents):
            if sender == ego:
                continue
            observations[ego, offset:offset + 2] = (
                estimated_states[ego, sender, :2] - ego_position
            )
            offset += 2
    return observations


def _estimated_voronoi_masks(learner, estimated_states, local_active):
    masks = []
    for ego, local_map in enumerate(learner.env.local_belief_maps):
        dones = np.asarray(local_active[ego]) < 0.5
        all_masks = local_map.get_voronoi_region_masks(
            estimated_states[ego, :, :2], dones
        )
        masks.append(all_masks[ego])
    return np.stack(masks).astype(bool)


def estimator_eval_act(
        self, observations, env_states, masks, goals, tasks,
        landmark_data, landmark_mask, deterministic=True):
    """Estimate motion, then delegate unchanged task/map logic to Stale path."""
    info = self.envs_info
    leaked = FORBIDDEN_POLICY_INFO_KEYS.intersection(info)
    if leaked:
        raise RuntimeError(
            "policy-facing info contains synchronized truth: {}".format(
                sorted(leaked)
            )
        )
    query_step = int(info["world_steps"])
    previous_step = getattr(self, "_estimator_query_step", None)
    if previous_step is None or query_step < previous_step or (
            query_step == 0 and previous_step != 0):
        self.state_estimator.reset(
            np.asarray(info["local_agent_states"], dtype=np.float32)[0],
            _initial_semantics(info),
            source_step=0,
        )
    self.state_estimator.ingest_deliveries(info.get("delivery_events", []))
    ego_states = _diagonal_rows(info["local_agent_states"])
    estimate = self.state_estimator.predict(
        query_step, ego_states, _current_semantics(info)
    )
    self._estimator_query_step = query_step
    self.last_estimator_output = estimate

    policy_info = dict(info)
    policy_info["local_agent_states"] = estimate.mean.copy()
    policy_info["local_voronoi_masks"] = _estimated_voronoi_masks(
        self, estimate.mean, info["local_agent_active"]
    )
    rewritten_observations = rewrite_low_level_teammates(
        observations, estimate.mean
    )

    original_info = self.envs_info
    self.envs_info = policy_info
    try:
        return stale_eval_act(
            self,
            rewritten_observations,
            env_states,
            masks,
            goals,
            tasks,
            landmark_data,
            landmark_mask,
            deterministic=deterministic,
        )
    finally:
        self.envs_info = original_info


def install_estimator_evaluator(learner, estimator):
    """Install an estimator without changing or training HIG-SAR weights."""
    learner.state_estimator = estimator
    learner._estimator_query_step = None
    learner.last_estimator_output = None
    learner.eval_act = types.MethodType(estimator_eval_act, learner)
    return learner


def build_estimator(method, num_agents, dt=0.1, checkpoint=None, device="cpu"):
    method = method.lower()
    if method == "stale":
        return StaleEstimator(num_agents, dt=dt)
    if method == "kf":
        if not checkpoint:
            raise ValueError("KF requires --estimator-checkpoint")
        return load_kf_checkpoint(checkpoint)
    if method == "commdrop":
        if not checkpoint:
            raise ValueError("CommDrop requires --estimator-checkpoint")
        return CommDropEstimator.from_checkpoint(checkpoint, device=device)
    raise ValueError("unknown estimator method {!r}".format(method))
