"""Trajectory storage, deterministic channel replay, and split validation."""

from __future__ import division

import hashlib
import os

import numpy as np
import torch

from multiagent.communication import (
    COMMUNICATION_PROFILES,
    CommunicationSimulator,
    get_communication_profile,
)


DATASET_VERSION = 1
TRAJECTORY_SEED = 7378
COMMUNICATION_SEED = 20260905


def make_episode_splits(num_episodes):
    """Return deterministic episode-index splits.

    The formal data set is exactly 700/150/150.  Small smoke data sets retain
    the same 70/15/15 proportions while keeping every split non-empty.
    """
    num_episodes = int(num_episodes)
    if num_episodes < 3:
        raise ValueError("at least three episodes are needed for train/val/test")
    if num_episodes == 1000:
        train_count, validation_count = 700, 150
    else:
        train_count = max(1, int(round(num_episodes * 0.70)))
        validation_count = max(1, int(round(num_episodes * 0.15)))
        if train_count + validation_count >= num_episodes:
            train_count = num_episodes - 2
            validation_count = 1
    indices = list(range(num_episodes))
    return {
        "train": indices[:train_count],
        "validation": indices[
            train_count:train_count + validation_count
        ],
        "test": indices[train_count + validation_count:],
    }


def validate_splits(splits, num_episodes, require_formal=False):
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("splits must contain exactly {}".format(sorted(required)))
    sets = {name: set(int(i) for i in values) for name, values in splits.items()}
    if any(sets[left].intersection(sets[right]) for left in required for right in required
           if left < right):
        raise ValueError("episode splits overlap")
    all_indices = sets["train"] | sets["validation"] | sets["test"]
    if all_indices != set(range(int(num_episodes))):
        raise ValueError("episode splits must cover every episode exactly once")
    if require_formal and [len(sets[name]) for name in (
            "train", "validation", "test")] != [700, 150, 150]:
        raise ValueError("formal data must use a 700/150/150 episode split")


def _validate_episode(episode, num_agents, episode_index):
    required = {
        "episode_index", "trajectory_seed", "communication_seed", "dt",
        "states", "goals", "options",
        "task_progress", "active", "landmarks",
    }
    missing = required.difference(episode)
    if missing:
        raise ValueError("episode is missing {}".format(sorted(missing)))
    if int(episode["episode_index"]) != episode_index:
        raise ValueError("episodes must be stored in episode-index order")
    states = np.asarray(episode["states"])
    if states.ndim != 3 or states.shape[1:] != (num_agents, 4):
        raise ValueError("states must have shape [time, agent, 4]")
    time_count = states.shape[0]
    expected = {
        "goals": (time_count, num_agents, 2),
        "options": (time_count, num_agents),
        "task_progress": (time_count, num_agents),
        "active": (time_count, num_agents),
        "landmarks": (num_agents, 2),
    }
    for key, shape in expected.items():
        if np.asarray(episode[key]).shape != shape:
            raise ValueError("{} must have shape {}".format(key, shape))


def validate_trajectory_dataset(dataset, require_formal=False):
    if int(dataset.get("version", -1)) != DATASET_VERSION:
        raise ValueError("unsupported estimator data-set version")
    episodes = dataset.get("episodes", [])
    num_agents = int(dataset.get("num_agents", 0))
    if num_agents < 1 or not episodes:
        raise ValueError("dataset must contain agents and episodes")
    validate_splits(dataset["splits"], len(episodes), require_formal=require_formal)
    trajectory_base = int(dataset.get("trajectory_seed_base", TRAJECTORY_SEED))
    communication_base = int(dataset.get(
        "communication_seed_base", COMMUNICATION_SEED
    ))
    if require_formal and (
            trajectory_base != TRAJECTORY_SEED
            or communication_base != COMMUNICATION_SEED):
        raise ValueError("formal data must use the fixed protocol seeds")
    for index, episode in enumerate(episodes):
        _validate_episode(episode, num_agents, index)
        if int(episode["trajectory_seed"]) != trajectory_base + index:
            raise ValueError("trajectory seed does not match dataset seed base")
        if int(episode.get("communication_seed", communication_base + index)) != communication_base + index:
            raise ValueError("communication seed does not match dataset seed base")
    return dataset


def save_trajectory_dataset(dataset, path):
    validate_trajectory_dataset(dataset, require_formal=False)
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    torch.save(dataset, temporary)
    os.replace(temporary, path)


def load_trajectory_dataset(path, require_formal=False):
    dataset = torch.load(
        os.path.abspath(path), map_location=lambda storage, location: storage
    )
    return validate_trajectory_dataset(dataset, require_formal=require_formal)


def _public_delivery(event, episode):
    payload = event["payload"]
    receiver = int(event["receiver"])
    source_step = int(event["source_step"])
    return {
        "arrival_step": int(event["arrival_step"]),
        "arrival_time": float(event["arrival_step"] * episode["dt"]),
        "receiver": receiver,
        "sender": int(event["sender"]),
        "source_step": source_step,
        "source_time": float(source_step * episode["dt"]),
        "motion_state": np.asarray(payload["motion_state"], dtype=np.float32).copy(),
        "option": int(payload["option"]),
        "goal": np.asarray(payload["goal"], dtype=np.float32).copy(),
        "task_progress": float(payload["task_progress"]),
        "active": float(payload["active"]),
        "ego_source_motion_state": np.asarray(
            episode["states"][source_step, receiver], dtype=np.float32
        ).copy(),
    }


def simulate_delivery_events(episode, level, communication_seed=None):
    """Replay a paper communication profile over one fixed ideal trajectory."""
    states = np.asarray(episode["states"], dtype=np.float32)
    num_steps, num_agents, _ = states.shape
    if communication_seed is None:
        communication_seed = int(episode.get(
            "communication_seed",
            COMMUNICATION_SEED + int(episode["episode_index"]),
        ))
    channel = CommunicationSimulator(
        num_agents,
        level=level,
        seed=int(communication_seed),
    )
    deliveries = [[] for _ in range(num_steps)]
    for query_step in range(1, num_steps):
        arrived = channel.deliver(query_step)
        for sequence, event in enumerate(arrived):
            public = _public_delivery(event, episode)
            public["sequence"] = int(sequence)
            deliveries[query_step].append(public)

        payloads = []
        for sender in range(num_agents):
            payloads.append({
                "motion_state": states[query_step, sender].copy(),
                "option": int(episode["options"][query_step, sender]),
                "goal": np.asarray(
                    episode["goals"][query_step, sender], dtype=np.float32
                ).copy(),
                "task_progress": float(
                    episode["task_progress"][query_step, sender]
                ),
                "active": float(episode["active"][query_step, sender]),
            })
        channel.schedule_broadcasts(
            query_step,
            states[query_step, :, :2],
            payloads,
        )
    return deliveries


def compute_normalization(episodes, epsilon=1e-6):
    states = np.concatenate([
        np.asarray(episode["states"], dtype=np.float64).reshape(-1, 4)
        for episode in episodes
    ], axis=0)
    goals = np.concatenate([
        np.asarray(episode["goals"], dtype=np.float64).reshape(-1, 2)
        for episode in episodes
    ], axis=0)
    progress = np.concatenate([
        np.asarray(episode["task_progress"], dtype=np.float64).reshape(-1, 1)
        for episode in episodes
    ], axis=0)

    def statistics(values):
        return {
            "mean": values.mean(axis=0).astype(np.float32),
            "std": np.maximum(values.std(axis=0), epsilon).astype(np.float32),
        }

    return {
        "state": statistics(states),
        "goal": statistics(goals),
        "task_progress": statistics(progress),
    }


def hash_policy_state_dicts(state_dicts):
    """Stable SHA-256 used to prove HIG-SAR stayed frozen."""
    digest = hashlib.sha256()
    for policy_index, state_dict in enumerate(state_dicts):
        digest.update(str(policy_index).encode("ascii"))
        for name in sorted(state_dict):
            value = state_dict[name]
            digest.update(name.encode("utf8"))
            if torch.is_tensor(value):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf8"))
    return digest.hexdigest()


def dataset_metadata(num_episodes, checkpoint, checkpoint_hash, dt=0.1):
    return {
        "version": DATASET_VERSION,
        "num_agents": 3,
        "dt": float(dt),
        "observation_radius": 0.5,
        "trajectory_seed_base": TRAJECTORY_SEED,
        "communication_seed_base": COMMUNICATION_SEED,
        "communication_profiles": {
            level: dict(profile)
            for level, profile in COMMUNICATION_PROFILES.items()
        },
        "policy_checkpoint": os.path.abspath(checkpoint),
        "policy_sha256": checkpoint_hash,
        "splits": make_episode_splits(num_episodes),
        "episodes": [],
    }
