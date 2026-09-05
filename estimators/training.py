"""KF calibration and episode-batched CommDrop training."""

from __future__ import division

import copy
import itertools
import os

import numpy as np
import torch

from estimators.base import StaleEstimator
from estimators.commdrop import CommDropEstimator, CommDropModel, commdrop_loss
from estimators.data import (
    COMMUNICATION_SEED,
    TRAJECTORY_SEED,
    compute_normalization,
    simulate_delivery_events,
)
from estimators.kalman import ConstantVelocityKalmanEstimator
from estimators.metrics import evaluate_estimator_episodes, summarize_episode_metrics


DEFAULT_OUTPUT_DIRECTORY = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "marlsave", "save_new", "chapter3_95_estimators_20260905",
))


def _atomic_torch_save(value, path):
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def _select_key(summary, secondary="position_rmse"):
    return (
        float(summary["spatial_edge_rmse"]["mean"]),
        float(summary[secondary]["mean"]),
    )


def calibrate_kf(dataset, level, output_path, multipliers=(0.25, 1.0, 4.0)):
    split = dataset["splits"]
    train_episodes = [dataset["episodes"][index] for index in split["train"]]
    validation_episodes = [
        dataset["episodes"][index] for index in split["validation"]
    ]
    base_q, base_r = ConstantVelocityKalmanEstimator.estimate_base_covariances(
        train_episodes, dt=float(dataset.get("dt", 0.1))
    )
    candidates = []
    for candidate in ConstantVelocityKalmanEstimator.covariance_grid(
            base_q, base_r, multipliers=multipliers):
        def factory(candidate=candidate):
            return ConstantVelocityKalmanEstimator(
                dataset["num_agents"],
                candidate["process_covariance"],
                candidate["observation_covariance"],
                dt=float(dataset.get("dt", 0.1)),
            )
        metrics = evaluate_estimator_episodes(
            factory,
            validation_episodes,
            level,
            simulate_delivery_events,
        )
        summary = summarize_episode_metrics(metrics)
        candidates.append((candidate, summary))
    best, best_summary = min(candidates, key=lambda item: _select_key(item[1]))
    checkpoint = {
        "kind": "kalman",
        "num_agents": int(dataset["num_agents"]),
        "dt": float(dataset.get("dt", 0.1)),
        "process_covariance": best["process_covariance"],
        "observation_covariance": best["observation_covariance"],
        "base_process_covariance": base_q,
        "base_observation_covariance": base_r,
        "q_multiplier": best["q_multiplier"],
        "r_multiplier": best["r_multiplier"],
        "communication_config": {
            "level": level,
            "profile": dict(dataset["communication_profiles"][level]),
        },
        "data_split": copy.deepcopy(split),
        "seeds": {
            "trajectory": int(dataset.get("trajectory_seed_base", TRAJECTORY_SEED)),
            "communication": int(dataset.get(
                "communication_seed_base", COMMUNICATION_SEED
            )),
        },
        "best_validation_epoch": None,
        "validation_metrics": best_summary,
        "policy_sha256": dataset.get("policy_sha256"),
    }
    _atomic_torch_save(checkpoint, output_path)
    return checkpoint


def load_kf_checkpoint(path):
    checkpoint = torch.load(
        os.path.abspath(path), map_location=lambda storage, location: storage
    )
    if checkpoint.get("kind") != "kalman":
        raise ValueError("not a Kalman estimator checkpoint")
    return ConstantVelocityKalmanEstimator(
        checkpoint["num_agents"],
        checkpoint["process_covariance"],
        checkpoint["observation_covariance"],
        dt=checkpoint["dt"],
    )


def _initial_training_memory(model, episode, device):
    states = torch.as_tensor(
        episode["states"][0], dtype=torch.float32, device=device
    )
    num_agents = states.shape[0]
    repeated_states = states.unsqueeze(0).expand(num_agents, -1, -1).clone()
    options = torch.as_tensor(
        episode["options"][0], dtype=torch.long, device=device
    ).unsqueeze(0).expand(num_agents, -1).clone()
    goals = torch.as_tensor(
        episode["goals"][0], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(num_agents, -1, -1).clone()
    progress = torch.as_tensor(
        episode["task_progress"][0], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(num_agents, -1).clone()
    active = torch.as_tensor(
        episode["active"][0], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(num_agents, -1).clone()
    source_steps = torch.zeros(num_agents, num_agents, device=device)
    initialized = torch.ones(
        num_agents, num_agents, dtype=torch.bool, device=device
    )
    previous = torch.zeros(
        num_agents * num_agents, model.hidden_dim, device=device
    )
    ego_source = states.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, 4)
    encoded = model.encode_event(
        repeated_states.reshape(-1, 4),
        ego_source,
        torch.zeros(num_agents * num_agents, device=device),
    )
    hidden = model.update_history(previous, encoded).reshape(
        num_agents, num_agents, model.hidden_dim
    )
    return {
        "hidden": hidden,
        "latest_events": encoded.reshape(
            num_agents, num_agents, model.hidden_dim
        ),
        "states": repeated_states,
        "options": options,
        "goals": goals,
        "progress": progress,
        "active": active,
        "source_steps": source_steps,
        "initialized": initialized,
    }


def _ingest_training_deliveries(model, memory, deliveries, device):
    for event in sorted(deliveries, key=lambda item: (
            int(item["arrival_step"]), int(item.get("sequence", 0)))):
        ego = int(event["receiver"])
        sender = int(event["sender"])
        source_step = int(event["source_step"])
        if source_step <= float(memory["source_steps"][ego, sender]):
            continue
        previous_step = float(memory["source_steps"][ego, sender])
        encoded = model.encode_event(
            torch.as_tensor(
                event["motion_state"], dtype=torch.float32, device=device
            ).reshape(1, 4),
            torch.as_tensor(
                event["ego_source_motion_state"],
                dtype=torch.float32, device=device,
            ).reshape(1, 4),
            torch.as_tensor([
                (source_step - previous_step) * model.config["dt"]
            ], device=device),
        )
        hidden = memory["hidden"].clone()
        hidden[ego, sender] = model.update_history(
            memory["hidden"][ego, sender].reshape(1, -1), encoded
        )[0]
        memory["hidden"] = hidden
        latest_events = memory["latest_events"].clone()
        latest_events[ego, sender] = encoded[0]
        memory["latest_events"] = latest_events
        for key, value in (
                ("states", event["motion_state"]),
                ("goals", event["goal"])):
            updated = memory[key].clone()
            updated[ego, sender] = torch.as_tensor(
                value, dtype=updated.dtype, device=device
            )
            memory[key] = updated
        for key, value in (
                ("options", event["option"]),
                ("progress", event["task_progress"]),
                ("active", event["active"]),
                ("source_steps", source_step)):
            updated = memory[key].clone()
            updated[ego, sender] = value
            memory[key] = updated
    return memory


def commdrop_episode_objective(model, episode, level, device):
    events = simulate_delivery_events(episode, level)
    memory = _initial_training_memory(model, episode, device)
    totals = None
    time_count = int(np.asarray(episode["states"]).shape[0])
    for query_step in range(time_count):
        memory = _ingest_training_deliveries(
            model, memory, events[query_step], device
        )
        truth = torch.as_tensor(
            episode["states"][query_step], dtype=torch.float32, device=device
        )
        options = memory["options"].clone()
        goals = memory["goals"].clone()
        progress = memory["progress"].clone()
        active = memory["active"].clone()
        indices = torch.arange(truth.shape[0], device=device)
        options[indices, indices] = torch.as_tensor(
            episode["options"][query_step], dtype=torch.long, device=device
        )
        goals[indices, indices] = torch.as_tensor(
            episode["goals"][query_step], dtype=torch.float32, device=device
        )
        progress[indices, indices] = torch.as_tensor(
            episode["task_progress"][query_step],
            dtype=torch.float32, device=device
        )
        active[indices, indices] = torch.as_tensor(
            episode["active"][query_step], dtype=torch.float32, device=device
        )
        prediction = model.predict_from_memory(
            memory["hidden"], memory["latest_events"], options, goals,
            progress, active, memory["source_steps"], memory["initialized"],
            query_step, truth,
        )
        losses = commdrop_loss(model, prediction, truth)
        if totals is None:
            totals = losses
        else:
            totals = {name: totals[name] + value for name, value in losses.items()}
    return {name: value / float(time_count) for name, value in totals.items()}


def _validation_summary(model, dataset, episodes, level, device):
    state = copy.deepcopy(model.state_dict())

    def factory():
        validation_model = CommDropModel(
            model.normalization_dict(), model.config
        )
        validation_model.load_state_dict(state)
        return CommDropEstimator(
            dataset["num_agents"], validation_model, device=device
        )
    metrics = evaluate_estimator_episodes(
        factory, episodes, level, simulate_delivery_events
    )
    return summarize_episode_metrics(metrics)


def train_commdrop(
        dataset, level, output_path, device="cpu", network_config=None,
        learning_rate=1e-3, weight_decay=1e-5, episode_batch_size=16,
        max_epochs=100, patience=15, gradient_clip=1.0, training_seed=20260905):
    torch.manual_seed(int(training_seed))
    np.random.seed(int(training_seed))
    split = dataset["splits"]
    train_episodes = [dataset["episodes"][index] for index in split["train"]]
    validation_episodes = [
        dataset["episodes"][index] for index in split["validation"]
    ]
    normalization = compute_normalization(train_episodes)
    config = dict(network_config or {})
    config["dt"] = float(dataset.get("dt", 0.1))
    model = CommDropModel(normalization, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_key = None
    best_state = None
    best_summary = None
    best_epoch = None
    stale_epochs = 0
    history = []

    for epoch in range(int(max_epochs)):
        model.train()
        order = np.random.RandomState(training_seed + epoch).permutation(
            len(train_episodes)
        )
        optimizer.zero_grad()
        batch_losses = []
        for position, episode_index in enumerate(order):
            losses = commdrop_episode_objective(
                model, train_episodes[int(episode_index)], level, device
            )
            (losses["loss"] / float(episode_batch_size)).backward()
            batch_losses.append(float(losses["loss"].detach().cpu()))
            is_boundary = (
                (position + 1) % int(episode_batch_size) == 0
                or position + 1 == len(order)
            )
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip)
                )
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        validation = _validation_summary(
            model, dataset, validation_episodes, level, device
        )
        key = _select_key(validation, secondary="nll")
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(batch_losses)),
            "validation": validation,
        })
        if best_key is None or key < best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_summary = validation
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break

    checkpoint = {
        "kind": "commdrop",
        "num_agents": int(dataset["num_agents"]),
        "model_state": best_state,
        "normalization": normalization,
        "network_config": copy.deepcopy(model.config),
        "communication_config": {
            "level": level,
            "profile": dict(dataset["communication_profiles"][level]),
        },
        "data_split": copy.deepcopy(split),
        "seeds": {
            "trajectory": int(dataset.get("trajectory_seed_base", TRAJECTORY_SEED)),
            "communication": int(dataset.get(
                "communication_seed_base", COMMUNICATION_SEED
            )),
            "training": int(training_seed),
        },
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "episode_batch_size": int(episode_batch_size),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "gradient_clip": float(gradient_clip),
        },
        "best_validation_epoch": int(best_epoch),
        "validation_metrics": best_summary,
        "training_history": history,
        "policy_sha256": dataset.get("policy_sha256"),
    }
    _atomic_torch_save(checkpoint, output_path)
    return checkpoint


def fallback_search_configs():
    """Validation-only search space allowed by the experiment protocol."""
    for hidden, learning_rate, sparse, kappa in itertools.product(
            (128, 256), (3e-4, 1e-3), (1e-4, 1e-3), (1.0, 3.0)):
        yield {
            "network_config": {
                "hidden_dim": hidden,
                "lambda_sparse": sparse,
                "kappa": kappa,
            },
            "learning_rate": learning_rate,
        }


def stale_validation(dataset, level):
    episodes = [
        dataset["episodes"][index]
        for index in dataset["splits"]["validation"]
    ]
    metrics = evaluate_estimator_episodes(
        lambda: StaleEstimator(
            dataset["num_agents"], dt=float(dataset.get("dt", 0.1))
        ),
        episodes,
        level,
        simulate_delivery_events,
    )
    return summarize_episode_metrics(metrics)
