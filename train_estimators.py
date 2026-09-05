"""Calibrate per-level KF and train per-level CommDrop checkpoints."""

from __future__ import print_function

import argparse
import os

import torch

from estimators.data import load_trajectory_dataset
from estimators.training import (
    DEFAULT_OUTPUT_DIRECTORY,
    calibrate_kf,
    fallback_search_configs,
    stale_validation,
    train_commdrop,
)


DEFAULT_DATASET = os.path.join(DEFAULT_OUTPUT_DIRECTORY, "ideal_trajectories.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--estimator", choices=("kf", "commdrop", "all"), default="all"
    )
    parser.add_argument(
        "--levels", nargs="+", choices=("mild", "medium", "severe"),
        default=("mild", "medium", "severe"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--training-seed", type=int, default=20260905)
    parser.add_argument("--no-fallback-search", action="store_true")
    parser.add_argument("--require-formal-dataset", action="store_true")
    return parser.parse_args()


def _metric(summary, name):
    return float(summary[name]["mean"])


def _train_one_commdrop(args, dataset, level, output_path, config, lr, seed):
    return train_commdrop(
        dataset,
        level,
        output_path,
        device=args.device,
        network_config=config,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        episode_batch_size=args.episode_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        training_seed=seed,
    )


def main():
    args = parse_args()
    dataset = load_trajectory_dataset(
        args.dataset, require_formal=args.require_formal_dataset
    )
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    kf_checkpoints = {}
    if args.estimator in ("kf", "all"):
        for level in args.levels:
            path = os.path.join(args.output_dir, "kf_{}.pt".format(level))
            print("Calibrating KF for {}".format(level))
            kf_checkpoints[level] = calibrate_kf(dataset, level, path)
            print(kf_checkpoints[level]["validation_metrics"])

    if args.estimator not in ("commdrop", "all"):
        return

    for level_index, level in enumerate(args.levels):
        kf_path = os.path.join(args.output_dir, "kf_{}.pt".format(level))
        if level not in kf_checkpoints:
            if not os.path.isfile(kf_path):
                print("KF validation is needed for comparison; calibrating {}".format(level))
                kf_checkpoints[level] = calibrate_kf(dataset, level, kf_path)
            else:
                kf_checkpoints[level] = torch.load(
                    kf_path, map_location=lambda storage, location: storage
                )
        stale_summary = stale_validation(dataset, level)
        final_path = os.path.join(
            args.output_dir, "commdrop_{}.pt".format(level)
        )
        default_config = {"hidden_dim": args.hidden_dim}
        print("Training default CommDrop for {}".format(level))
        best = _train_one_commdrop(
            args, dataset, level, final_path, default_config,
            args.learning_rate, args.training_seed + level_index,
        )
        comm_edge = _metric(best["validation_metrics"], "spatial_edge_rmse")
        stale_edge = _metric(stale_summary, "spatial_edge_rmse")
        kf_edge = _metric(
            kf_checkpoints[level]["validation_metrics"], "spatial_edge_rmse"
        )
        needs_search = comm_edge >= stale_edge or comm_edge >= kf_edge
        if needs_search and not args.no_fallback_search:
            print(
                "Default did not beat both validation baselines; starting the "
                "protocol-limited search for {}".format(level)
            )
            search_dir = os.path.join(args.output_dir, "validation_search", level)
            if not os.path.isdir(search_dir):
                os.makedirs(search_dir)
            candidates = [best]
            for candidate_index, candidate in enumerate(fallback_search_configs()):
                config = candidate["network_config"]
                lr = candidate["learning_rate"]
                if (
                    config["hidden_dim"] == args.hidden_dim
                    and config["lambda_sparse"] == 1e-3
                    and config["kappa"] == 3.0
                    and lr == args.learning_rate
                ):
                    continue
                candidate_path = os.path.join(
                    search_dir, "candidate_{:02d}.pt".format(candidate_index)
                )
                trained = _train_one_commdrop(
                    args, dataset, level, candidate_path, config, lr,
                    args.training_seed + 100 * (level_index + 1) + candidate_index,
                )
                candidates.append(trained)
            best = min(candidates, key=lambda checkpoint: (
                _metric(checkpoint["validation_metrics"], "spatial_edge_rmse"),
                _metric(checkpoint["validation_metrics"], "nll"),
            ))
            temporary = final_path + ".tmp"
            torch.save(best, temporary)
            os.replace(temporary, final_path)
        print("Selected {} validation metrics: {}".format(
            level, best["validation_metrics"]
        ))


if __name__ == "__main__":
    main()
