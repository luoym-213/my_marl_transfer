"""Validation-gated offline and paired closed-loop estimator evaluation."""

from __future__ import print_function

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from communication_baseline import make_ideal_env
from estimator_communication import make_estimator_communication_env
from estimator_policy import build_estimator, install_estimator_evaluator
from estimators.commdrop import CommDropEstimator
from estimators.data import (
    COMMUNICATION_SEED,
    TRAJECTORY_SEED,
    hash_policy_state_dicts,
    load_trajectory_dataset,
    simulate_delivery_events,
)
from estimators.metrics import (
    EpisodeMetricAccumulator,
    evaluate_estimator_episodes,
    summarize_episode_metrics,
)
from estimators.training import (
    DEFAULT_OUTPUT_DIRECTORY,
    load_kf_checkpoint,
)
from learner import setup_master
from utils import normalize_obs


REPOSITORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(DEFAULT_OUTPUT_DIRECTORY, "ideal_trajectories.pt")
DEFAULT_POLICY_CHECKPOINT = os.path.abspath(os.path.join(
    REPOSITORY,
    "..", "marlsave", "save_new",
    "chapter2_94_attn_fallback_20260905_05", "ep400.pt",
))


def _has_option(arguments, option):
    return option in arguments or any(
        argument.startswith(option + "=") for argument in arguments
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("offline", "closed-loop", "both"), default="both")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--policy-checkpoint", default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--results-json", default=os.path.join(
        REPOSITORY, "chapter3_95_estimator_results.json"
    ))
    parser.add_argument("--results-markdown", default=os.path.join(
        REPOSITORY, "CHAPTER3_95_ESTIMATOR_RESULTS.md"
    ))
    parser.add_argument("--num-closed-loop-episodes", type=int, default=150)
    parser.add_argument("--comm-seed", type=int, default=COMMUNICATION_SEED)
    parser.add_argument("--require-formal-dataset", action="store_true")
    known, remaining = parser.parse_known_args()
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
    for name, value in vars(known).items():
        setattr(args, name, value)
    return args


def _checkpoint_path(output_directory, method, level):
    return os.path.join(output_directory, "{}_{}.pt".format(method, level))


def _offline_factory(method, dataset, output_directory, level, device):
    if method == "stale":
        from estimators.base import StaleEstimator
        return lambda: StaleEstimator(
            dataset["num_agents"], dt=float(dataset.get("dt", 0.1))
        )
    path = _checkpoint_path(output_directory, method, level)
    if not os.path.isfile(path):
        raise IOError("missing {} checkpoint: {}".format(method, path))
    if method == "kf":
        return lambda: load_kf_checkpoint(path)
    if method == "commdrop":
        return lambda: CommDropEstimator.from_checkpoint(path, device=device)
    raise ValueError(method)


def offline_split(dataset, split_name, output_directory, device):
    episodes = [
        dataset["episodes"][index] for index in dataset["splits"][split_name]
    ]
    result = {}
    for level in ("mild", "medium", "severe"):
        result[level] = {}
        for method in ("stale", "kf", "commdrop"):
            factory = _offline_factory(
                method, dataset, output_directory, level, device
            )
            metrics = evaluate_estimator_episodes(
                factory, episodes, level, simulate_delivery_events
            )
            result[level][method] = summarize_episode_metrics(metrics)
    return result


def validation_gate(validation):
    per_level_stale = {}
    commdrop_edges = []
    kalman_edges = []
    for level in ("mild", "medium", "severe"):
        commdrop = validation[level]["commdrop"]["spatial_edge_rmse"]["mean"]
        stale = validation[level]["stale"]["spatial_edge_rmse"]["mean"]
        kalman = validation[level]["kf"]["spatial_edge_rmse"]["mean"]
        per_level_stale[level] = bool(commdrop < stale)
        commdrop_edges.append(float(commdrop))
        kalman_edges.append(float(kalman))
    macro_commdrop = float(np.mean(commdrop_edges))
    macro_kalman = float(np.mean(kalman_edges))
    passed = all(per_level_stale.values()) and macro_commdrop < macro_kalman
    return {
        "passed": bool(passed),
        "commdrop_below_stale_by_level": per_level_stale,
        "commdrop_macro_spatial_edge_rmse": macro_commdrop,
        "kf_macro_spatial_edge_rmse": macro_kalman,
        "commdrop_macro_below_kf": bool(macro_commdrop < macro_kalman),
    }


def _scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(float(finite.size))
    return {
        "mean": mean, "std": std,
        "ci95_low": mean - half_width, "ci95_high": mean + half_width,
    }


def _discovery_updates(env, step, discovered):
    agent_positions = np.asarray(
        [agent.state.p_pos for agent in env.agents], dtype=np.float32
    )
    landmarks = np.asarray(
        [landmark.state.p_pos for landmark in env.world.landmarks],
        dtype=np.float32,
    )
    visible = np.linalg.norm(
        agent_positions[:, None, :] - landmarks[None, :, :], axis=-1
    ).min(axis=0) <= float(env.world.mask_obs_dist)
    for index in np.flatnonzero(visible):
        discovered.setdefault(int(index), int(step))


def _condition_env(args, condition, method):
    if condition == "ideal":
        env = make_ideal_env(
            args.env_name, args.num_agents, args.dist_threshold,
            args.arena_size, args.identity_size,
            mask_obs_dist=0.5, trajectory_seed=args.seed,
        )
        return env, setup_master(args, env=env), None
    env = make_estimator_communication_env(
        args.env_name, args.num_agents, args.dist_threshold,
        args.arena_size, args.identity_size,
        mask_obs_dist=0.5, communication_level=condition,
        communication_seed=args.comm_seed,
        trajectory_seed=args.seed,
    )
    master = setup_master(args, env=env)
    checkpoint = None if method == "stale" else _checkpoint_path(
        args.output_dir, method, condition
    )
    estimator = build_estimator(
        method, args.num_agents, dt=float(env.world.dt),
        checkpoint=checkpoint, device=args.device,
    )
    install_estimator_evaluator(master, estimator)
    return env, master, estimator


def closed_loop_condition(args, policies, observation_stats, condition, method):
    env, master, estimator = _condition_env(args, condition, method)
    master.load_models(policies)
    master.set_eval_mode()
    before_hash = hash_policy_state_dicts(master.all_policies)
    observation_mean, observation_std = observation_stats
    episodes = []
    estimator_metrics = []
    for episode_index in range(int(args.num_closed_loop_episodes)):
        observations, env_state, info = env.reset()
        master.set_envs_info(info)
        observations = normalize_obs(observations, observation_mean, observation_std)
        masks = torch.ones(env.n, 1, device=args.device)
        goals = torch.zeros(env.n, 2, dtype=torch.float32, device=args.device)
        options = torch.zeros(env.n, 1, dtype=torch.long, device=args.device)
        landmark_data = torch.zeros(
            env.n, env.n, 4, dtype=torch.float32, device=args.device
        )
        landmark_mask = torch.zeros(
            env.n, env.n, 1, dtype=torch.float32, device=args.device
        )
        done = np.zeros(env.n, dtype=bool)
        discovered = {}
        rescued = set()
        response_times = []
        _discovery_updates(env, 0, discovered)
        accumulator = EpisodeMetricAccumulator(env.n) if estimator else None
        while not bool(done.all()):
            with torch.no_grad():
                actions, goals, options, landmark_data, landmark_mask = master.eval_act(
                    observations, env_state, masks, goals, options,
                    landmark_data, landmark_mask, deterministic=True,
                )
            if estimator is not None:
                truth = np.asarray([
                    np.concatenate([agent.state.p_pos, agent.state.p_vel])
                    for agent in env.agents
                ], dtype=np.float32)
                accumulator.add(master.last_estimator_output, truth, truth)
            goals_array = goals.detach().cpu().numpy()
            options_array = options.detach().cpu().numpy()
            observations, _, _, done_info, info, env_state = env.step({
                "agents_actions": actions,
                "agents_goals": goals_array,
                "agents_tasks": options_array,
            })
            done = np.asarray(done_info["agent"], dtype=bool)
            masks = torch.as_tensor((~done).astype(np.float32), device=args.device)
            observations = normalize_obs(
                observations, observation_mean, observation_std
            )
            master.set_envs_info(info)
            step = int(info["world_steps"])
            _discovery_updates(env, step, discovered)
            for landmark_index in set(env.visited_landmarks).difference(rescued):
                rescued.add(int(landmark_index))
                if int(landmark_index) in discovered:
                    response_times.append(
                        (step - discovered[int(landmark_index)]) * env.world.dt
                    )
        final_distance = float(np.asarray(env.world.min_dists).mean())
        success = bool(info["is_success"])
        episodes.append({
            "episode_index": episode_index,
            "trajectory_seed": args.seed + episode_index,
            "communication_seed": (
                None if condition == "ideal"
                else args.comm_seed + episode_index
            ),
            "success": success,
            "completion_time": (
                float(info["world_steps"] * env.world.dt) if success else None
            ),
            "discovery_to_rescue_time": (
                float(np.mean(response_times)) if response_times else None
            ),
            "final_distance": final_distance,
        })
        if accumulator is not None:
            estimator_metrics.append(accumulator.result())
    after_hash = hash_policy_state_dicts(master.all_policies)
    if after_hash != before_hash:
        raise RuntimeError("HIG-SAR changed during closed-loop evaluation")
    successes = [float(item["success"]) for item in episodes]
    completion = [
        item["completion_time"] for item in episodes
        if item["completion_time"] is not None
    ]
    response = [
        item["discovery_to_rescue_time"] for item in episodes
        if item["discovery_to_rescue_time"] is not None
    ]
    summary = {
        "episodes": len(episodes),
        "success_rate": _scalar_summary(successes),
        "successful_completion_time": _scalar_summary(completion),
        "discovery_to_rescue_time": _scalar_summary(response),
        "final_distance": _scalar_summary([
            item["final_distance"] for item in episodes
        ]),
        "policy_sha256_before": before_hash,
        "policy_sha256_after": after_hash,
    }
    if estimator_metrics:
        summary["state_estimation"] = summarize_episode_metrics(estimator_metrics)
    else:
        zero = {
            "mean": 0.0, "std": 0.0,
            "ci95_low": 0.0, "ci95_high": 0.0,
            "episodes": len(episodes),
        }
        summary["state_estimation"] = {
            "position_rmse": dict(zero),
            "velocity_rmse": dict(zero),
            "spatial_edge_rmse": dict(zero),
            "nll": {
                "mean": None, "std": None, "ci95_low": None,
                "ci95_high": None, "episodes": len(episodes),
            },
        }
    return summary


def closed_loop(args):
    if not os.path.isfile(args.policy_checkpoint):
        raise IOError("missing HIG-SAR checkpoint: {}".format(
            args.policy_checkpoint
        ))
    checkpoint = torch.load(
        args.policy_checkpoint, map_location=lambda storage, location: storage
    )
    policies = checkpoint["models"]
    observation_stats = checkpoint.get("ob_rms", (None, None))
    result = {
        "ideal": closed_loop_condition(
            args, policies, observation_stats, "ideal", None
        )
    }
    for level in ("mild", "medium", "severe"):
        result[level] = {}
        for method in ("stale", "kf", "commdrop"):
            print("Closed loop: {} {}".format(level, method))
            result[level][method] = closed_loop_condition(
                args, policies, observation_stats, level, method
            )
    return result


def _write_results(result, json_path, markdown_path):
    for path in (json_path, markdown_path):
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    with open(json_path, "w") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
    lines = [
        "# Chapter 3 state-estimator results",
        "",
        "Validation gate: **{}**".format(
            "PASS" if result["validation_gate"]["passed"] else "FAIL"
        ),
        "",
    ]
    if "offline_test" in result:
        lines.extend([
            "## Offline test",
            "",
            "| Level | Method | Position RMSE | Velocity RMSE | Spatial-edge RMSE | NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for level in ("mild", "medium", "severe"):
            for method in ("stale", "kf", "commdrop"):
                metrics = result["offline_test"][level][method]
                lines.append(
                    "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                        level, method,
                        metrics["position_rmse"]["mean"],
                        metrics["velocity_rmse"]["mean"],
                        metrics["spatial_edge_rmse"]["mean"],
                        metrics["nll"]["mean"],
                    )
                )
        lines.append("")
    if "closed_loop" in result:
        lines.extend([
            "## Paired closed-loop",
            "",
            "| Level | Method | Success | Successful time | Response time | Final distance | Position RMSE | Velocity RMSE | Spatial-edge RMSE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        rows = [("ideal", "ideal", result["closed_loop"]["ideal"])]
        for level in ("mild", "medium", "severe"):
            for method in ("stale", "kf", "commdrop"):
                rows.append((level, method, result["closed_loop"][level][method]))
        for level, method, metrics in rows:
            state = metrics["state_estimation"]
            values = [
                metrics["success_rate"]["mean"],
                metrics["successful_completion_time"]["mean"],
                metrics["discovery_to_rescue_time"]["mean"],
                metrics["final_distance"]["mean"],
                state["position_rmse"]["mean"],
                state["velocity_rmse"]["mean"],
                state["spatial_edge_rmse"]["mean"],
            ]
            rendered = [
                "NA" if value is None else "{:.6f}".format(value)
                for value in values
            ]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                level, method, *rendered
            ))
        lines.append("")
    if not result["validation_gate"]["passed"]:
        lines.extend([
            "The test split and closed-loop evaluation remain sealed because ",
            "the validation criterion was not met.",
            "",
        ])
    with open(markdown_path, "w") as output:
        output.write("\n".join(lines))


def main():
    args = parse_args()
    dataset = load_trajectory_dataset(
        args.dataset, require_formal=args.require_formal_dataset
    )
    result = {
        "protocol": {
            "trajectory_seed": int(args.seed),
            "communication_seed": int(args.comm_seed),
            "observation_radius": 0.5,
            "policy_checkpoint": os.path.abspath(args.policy_checkpoint),
            "dataset": os.path.abspath(args.dataset),
        }
    }
    print("Evaluating sealed validation split")
    validation = offline_split(
        dataset, "validation", args.output_dir, args.device
    )
    result["offline_validation"] = validation
    result["validation_gate"] = validation_gate(validation)
    if result["validation_gate"]["passed"]:
        if args.mode in ("offline", "both"):
            print("Validation passed; opening offline test split")
            result["offline_test"] = offline_split(
                dataset, "test", args.output_dir, args.device
            )
        if args.mode in ("closed-loop", "both"):
            print("Validation passed; starting paired closed-loop evaluation")
            result["closed_loop"] = closed_loop(args)
    else:
        print("Validation gate failed; test and closed-loop results remain sealed")
    _write_results(result, args.results_json, args.results_markdown)
    print(json.dumps(result["validation_gate"], ensure_ascii=False, indent=2))
    print("Saved results to {} and {}".format(
        os.path.abspath(args.results_json), os.path.abspath(args.results_markdown)
    ))


if __name__ == "__main__":
    main()
