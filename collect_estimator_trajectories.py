"""Collect ideal-communication trajectories from the frozen 0.5 HIG-SAR."""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch

from communication_baseline import make_ideal_env
from estimators.data import (
    COMMUNICATION_SEED,
    TRAJECTORY_SEED,
    dataset_metadata,
    hash_policy_state_dicts,
    save_trajectory_dataset,
)
from estimators.training import DEFAULT_OUTPUT_DIRECTORY
from learner import setup_master
from utils import normalize_obs


REPOSITORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_CHECKPOINT = os.path.abspath(os.path.join(
    REPOSITORY,
    "..", "marlsave", "save_new",
    "chapter2_94_attn_fallback_20260905_05", "ep400.pt",
))
DEFAULT_DATASET = os.path.join(DEFAULT_OUTPUT_DIRECTORY, "ideal_trajectories.pt")


def _has_option(arguments, option):
    return option in arguments or any(
        argument.startswith(option + "=") for argument in arguments
    )


def parse_args():
    collector = argparse.ArgumentParser(add_help=False)
    collector.add_argument("--episodes", type=int, default=1000)
    collector.add_argument("--dataset-path", default=DEFAULT_DATASET)
    collector.add_argument("--checkpoint", default=DEFAULT_POLICY_CHECKPOINT)
    collector.add_argument("--comm-seed", type=int, default=COMMUNICATION_SEED)
    known, remaining = collector.parse_known_args()
    defaults = {
        "--mask-obs-dist": "0.5",
        "--rrt-max-iter": "50",
        "--seed": str(TRAJECTORY_SEED),
    }
    for option, value in defaults.items():
        if not _has_option(remaining, option):
            remaining.extend([option, value])
    for flag in ("--entity-mp", "--test"):
        if flag not in remaining:
            remaining.append(flag)
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        from arguments import get_args
        args = get_args()
    finally:
        sys.argv = original
    args.episodes = known.episodes
    args.dataset_path = known.dataset_path
    args.checkpoint = known.checkpoint
    args.comm_seed = known.comm_seed
    return args


def _motion_states(env):
    return np.asarray([
        np.concatenate([agent.state.p_pos, agent.state.p_vel])
        for agent in env.agents
    ], dtype=np.float32)


def _append_frame(storage, env, goals, options):
    states = _motion_states(env)
    goals = np.asarray(goals, dtype=np.float32).reshape(env.n, 2)
    options = np.asarray(options, dtype=np.int64).reshape(env.n)
    storage["states"].append(states)
    storage["goals"].append(goals.copy())
    storage["options"].append(options.copy())
    storage["task_progress"].append(
        np.linalg.norm(goals - states[:, :2], axis=1).astype(np.float32)
    )
    storage["active"].append(
        (~np.asarray(env.agents_done, dtype=bool)).astype(np.float32)
    )


def collect(args):
    if int(args.episodes) < 3:
        raise ValueError("--episodes must be at least 3")
    if not os.path.isfile(args.checkpoint):
        raise IOError("HIG-SAR checkpoint does not exist: {}".format(
            args.checkpoint
        ))
    checkpoint = torch.load(
        args.checkpoint, map_location=lambda storage, location: storage
    )
    policies = checkpoint["models"]
    policy_hash = hash_policy_state_dicts(policies)
    observation_stats = checkpoint.get("ob_rms", (None, None))

    env = make_ideal_env(
        args.env_name,
        args.num_agents,
        args.dist_threshold,
        args.arena_size,
        args.identity_size,
        mask_obs_dist=0.5,
        trajectory_seed=args.seed,
    )
    master = setup_master(args, env=env)
    master.load_models(policies)
    master.set_eval_mode()
    for agent in master.all_agents:
        for parameter in agent.actor_critic.parameters():
            parameter.requires_grad_(False)

    dataset = dataset_metadata(
        args.episodes, args.checkpoint, policy_hash, dt=float(env.world.dt)
    )
    dataset["num_agents"] = int(env.n)
    dataset["trajectory_seed_base"] = int(args.seed)
    dataset["communication_seed_base"] = int(args.comm_seed)
    observation_mean, observation_std = observation_stats
    for episode_index in range(int(args.episodes)):
        observations, env_state, info = env.reset()
        master.set_envs_info(info)
        observations = normalize_obs(
            observations, observation_mean, observation_std
        )
        masks = torch.ones(env.n, 1, device=args.device)
        initial_positions = np.asarray(
            [agent.state.p_pos for agent in env.agents], dtype=np.float32
        )
        goals = torch.as_tensor(
            initial_positions, dtype=torch.float32, device=args.device
        )
        options = torch.zeros(
            env.n, 1, dtype=torch.long, device=args.device
        )
        landmark_data = torch.zeros(
            env.n, env.n, 4, dtype=torch.float32, device=args.device
        )
        landmark_mask = torch.zeros(
            env.n, env.n, 1, dtype=torch.float32, device=args.device
        )
        frames = {
            "states": [], "goals": [], "options": [],
            "task_progress": [], "active": [],
        }
        _append_frame(frames, env, initial_positions, np.zeros(env.n))
        done = np.zeros(env.n, dtype=bool)
        while not bool(done.all()):
            with torch.no_grad():
                actions, goals, options, landmark_data, landmark_mask = (
                    master.eval_act(
                        observations, env_state, masks, goals, options,
                        landmark_data, landmark_mask, deterministic=True,
                    )
                )
            goals_array = goals.detach().cpu().numpy()
            options_array = options.detach().cpu().numpy()
            step_data = {
                "agents_actions": actions,
                "agents_goals": goals_array,
                "agents_tasks": options_array,
            }
            observations, _, _, done_info, info, env_state = env.step(step_data)
            done = np.asarray(done_info["agent"], dtype=bool)
            masks = torch.as_tensor(
                (~done).astype(np.float32), device=args.device
            )
            observations = normalize_obs(
                observations, observation_mean, observation_std
            )
            master.set_envs_info(info)
            _append_frame(frames, env, goals_array, options_array)

        episode = {
            "episode_index": int(episode_index),
            "trajectory_seed": int(args.seed + episode_index),
            "communication_seed": int(args.comm_seed + episode_index),
            "dt": float(env.world.dt),
            "landmarks": np.asarray(
                [landmark.state.p_pos for landmark in env.world.landmarks],
                dtype=np.float32,
            ),
        }
        for name, values in frames.items():
            episode[name] = np.asarray(values)
        dataset["episodes"].append(episode)
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == args.episodes:
            print("Collected {}/{} episodes".format(
                episode_index + 1, args.episodes
            ))

    after_hash = hash_policy_state_dicts(master.all_policies)
    if after_hash != policy_hash:
        raise RuntimeError("frozen HIG-SAR weights changed during collection")
    dataset["policy_sha256_after_collection"] = after_hash
    save_trajectory_dataset(dataset, args.dataset_path)
    print("Saved {} trajectories to {}".format(
        len(dataset["episodes"]), os.path.abspath(args.dataset_path)
    ))
    print("Frozen HIG-SAR SHA-256: {}".format(policy_hash))


def main():
    collect(parse_args())


if __name__ == "__main__":
    main()
