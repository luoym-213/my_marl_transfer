"""Evaluate the frozen HIG-SAR policy under paper-defined communication levels.

Example:
    PYTHONPATH=src/multiagent/mape:src/gym-vecenv \
      python eval_comm.py --no-cuda

The default checkpoint is the latest observation-radius-0.5 checkpoint requested
for Chapter 3.  Command-line options accepted by ``eval.py`` remain available.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.abspath(os.path.join(
    REPO_DIR,
    "..",
    "marlsave",
    "save_new",
    "chapter2_94_attn_fallback_20260905_05",
    "ep400.pt",
))


def _has_option(arguments, option):
    return option in arguments or any(
        argument.startswith(option + "=") for argument in arguments
    )


def _parse_args():
    communication_parser = argparse.ArgumentParser(add_help=False)
    communication_parser.add_argument("--comm-seed", type=int, default=20260905)
    communication_parser.add_argument("--comm-results-path", default=None)
    communication_args, remaining = communication_parser.parse_known_args()

    defaults = {
        "--load-dir": DEFAULT_CHECKPOINT,
        "--mask-obs-dist": "0.5",
        "--rrt-max-iter": "50",
        "--num-eval-episodes": "50",
        "--seed": "7378",
    }
    for option, value in defaults.items():
        if not _has_option(remaining, option):
            remaining.extend([option, value])
    if "--entity-mp" not in remaining:
        remaining.append("--entity-mp")
    if "--test" not in remaining:
        remaining.append("--test")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        from arguments import get_args
        args = get_args()
    finally:
        sys.argv = original_argv
    args.comm_seed = communication_args.comm_seed
    args.comm_results_path = communication_args.comm_results_path
    return args


def _summarize(level, results, num_episodes):
    (
        per_episode_rewards,
        per_step_rewards,
        per_episode_high_rewards,
        per_high_step_rewards,
        final_min_dists,
        num_success,
        episode_length,
        successful_average_length,
        successful_count,
    ) = results
    del per_episode_rewards, per_episode_high_rewards
    final_distance = (
        float(np.stack(final_min_dists).mean()) if final_min_dists else None
    )
    return {
        "condition": level,
        "episodes": int(num_episodes),
        "successes": int(num_success),
        "success_rate": float(num_success) / float(num_episodes),
        "mean_episode_steps": float(episode_length),
        "mean_success_steps": (
            float(successful_average_length) if successful_count else None
        ),
        "mean_per_step_reward": float(per_step_rewards.mean()),
        "mean_high_per_step_reward": float(per_high_step_rewards.mean()),
        "mean_final_distance": final_distance,
    }


def _evaluate_condition(args, policies, ob_rms, level):
    from eval import evaluate
    from learner import setup_master
    from communication_baseline import (
        install_stale_evaluator,
        make_communication_env,
        make_ideal_env,
    )

    condition_args = copy.copy(args)
    if level == "ideal":
        env = make_ideal_env(
            condition_args.env_name,
            condition_args.num_agents,
            condition_args.dist_threshold,
            condition_args.arena_size,
            condition_args.identity_size,
            mask_obs_dist=condition_args.mask_obs_dist,
            trajectory_seed=condition_args.seed,
        )
        master = setup_master(condition_args, env=env)
    else:
        env = make_communication_env(
            condition_args.env_name,
            condition_args.num_agents,
            condition_args.dist_threshold,
            condition_args.arena_size,
            condition_args.identity_size,
            mask_obs_dist=condition_args.mask_obs_dist,
            communication_level=level,
            communication_seed=condition_args.comm_seed,
            trajectory_seed=condition_args.seed,
        )
        master = setup_master(condition_args, env=env)
        install_stale_evaluator(master)

    return evaluate(
        condition_args,
        condition_args.seed,
        policies,
        ob_rms=ob_rms,
        render=condition_args.render,
        env=env,
        master=master,
        render_attn=False,
    )


def main():
    args = _parse_args()
    if not os.path.isfile(args.load_dir):
        raise IOError("Checkpoint does not exist: {}".format(args.load_dir))

    checkpoint = torch.load(
        args.load_dir, map_location=lambda storage, location: storage
    )
    policies = checkpoint["models"]
    ob_rms = checkpoint.get("ob_rms", (None, None))

    summaries = []
    for level in ("ideal", "mild", "medium", "severe"):
        print("\n===== Evaluating {} =====".format(level))
        results = _evaluate_condition(args, policies, ob_rms, level)
        summaries.append(_summarize(
            level, results, args.num_eval_episodes
        ))

    print("\nCommunication baseline summary")
    print("condition  success   success_steps  final_distance  high_reward/step")
    for row in summaries:
        success_steps = row["mean_success_steps"]
        print(
            "{:<9} {:>6.2%} {:>15} {:>15.4f} {:>17.4f}".format(
                row["condition"],
                row["success_rate"],
                "NA" if success_steps is None else "{:.2f}".format(success_steps),
                row["mean_final_distance"],
                row["mean_high_per_step_reward"],
            )
        )

    output = {
        "checkpoint": os.path.abspath(args.load_dir),
        "trajectory_seed": int(args.seed),
        "communication_seed": int(args.comm_seed),
        "mask_observation_radius": float(args.mask_obs_dist),
        "results": summaries,
    }
    print("\n" + json.dumps(output, ensure_ascii=False, indent=2))
    if args.comm_results_path:
        result_path = os.path.abspath(args.comm_results_path)
        with open(result_path, "w") as result_file:
            json.dump(output, result_file, ensure_ascii=False, indent=2)
        print("Saved results to {}".format(result_path))


if __name__ == "__main__":
    main()
