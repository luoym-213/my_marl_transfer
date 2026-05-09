import copy
import os
import sys
from argparse import Namespace

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from gym.spaces import Discrete

from marl.algos.ppo import feed_forward_generator
from marl.algos.storage import RolloutStorage
from marl.agents.learner import setup_master
from marl.controllers.landmark_memory import LandmarkMemory
from marl.controllers.local_maps import (
    LocalMapBank,
    build_comm_mask,
    build_low_level_comm_features,
    flatten_agent_major,
)
from marl.utils import make_parallel_envs


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _base_args(num_processes=2):
    return Namespace(
        env_name="simple_spread",
        num_agents=3,
        masking=False,
        mask_dist=None,
        mask_obs_dist=0.3,
        comm_dist=0.5,
        sensor_dist=0.3,
        local_voronoi_scope="comm",
        dropout_masking=False,
        entity_mp=True,
        identity_size=0,
        rrt_max_iter=5,
        top_k=3,
        seed=123,
        num_processes=num_processes,
        num_steps=4,
        no_cuda=True,
        gpu_id=None,
        num_frames=8,
        arena_size=1,
        high_level_interval=5,
        num_eval_episodes=1,
        dist_threshold=0.1,
        render=False,
        record_video=False,
        gif_save_path="gifs",
        algo="ppo",
        lr=1e-4,
        gamma=0.99,
        tau=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        ppo_epoch=1,
        num_mini_batch=1,
        clip_param=0.2,
        recurrent_hidden_state_size=128,
        is_recurrent=False,
        load_low_level_path=None,
        load_high_level_path=None,
        load_high_critic_path=None,
        save_dir="smoke_test",
        log_dir="logs",
        save_interval=200,
        log_interval=10,
        test=True,
        load_dir=None,
        eval_interval=None,
        continue_training=False,
        branch="validation",
        no_clipped_value_loss=False,
        clipped_value_loss=True,
        cuda=False,
        device=torch.device("cpu"),
    )


def test_local_map_timestamp_fusion():
    bank = LocalMapBank(2, 3, "cpu")
    positions = torch.tensor(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.9, 0.0]],
            [[0.0, 0.0], [0.6, 0.0], [0.0, 0.6]],
        ],
        dtype=torch.float32,
    )
    detections = [
        [torch.tensor([[0.01, 0.0]]), torch.zeros(0, 2), torch.zeros(0, 2)],
        [torch.zeros(0, 2), torch.tensor([[0.6, 0.0]]), torch.zeros(0, 2)],
    ]
    bank.update_from_local_observations(positions, detections, torch.tensor([1.0, 1.0]), 0.3)
    before_no_comm = bank.belief_maps.clone()
    no_comm = torch.eye(3, dtype=torch.bool).view(1, 3, 3).repeat(2, 1, 1)
    bank.fuse_by_timestamp(no_comm)
    _assert(torch.equal(bank.belief_maps, before_no_comm), "No-comm fusion changed maps")

    cell = bank.world_to_grid(torch.tensor([0.01, 0.0]))
    bank.belief_maps[0, 0, cell[0], cell[1]] = 0.9
    bank.timestamp_maps[0, 0, cell[0], cell[1]] = 5.0
    bank.belief_maps[0, 1, cell[0], cell[1]] = 0.1
    bank.timestamp_maps[0, 1, cell[0], cell[1]] = 3.0
    comm = no_comm.clone()
    comm[0, 1, 0] = True
    bank.fuse_by_timestamp(comm)
    _assert(
        torch.isclose(bank.belief_maps[0, 1, cell[0], cell[1]], torch.tensor(0.9)),
        "Newer timestamp did not overwrite receiver cell",
    )

    bank.belief_maps[0, 1, cell[0], cell[1]] = 0.2
    bank.timestamp_maps[0, 1, cell[0], cell[1]] = 5.0
    bank.fuse_by_timestamp(comm)
    _assert(
        torch.isclose(bank.belief_maps[0, 1, cell[0], cell[1]], torch.tensor(0.2)),
        "Equal timestamp should keep receiver value",
    )

    bank.belief_maps[0].fill_(0.7)
    bank.timestamp_maps[0].fill_(9.0)
    bank.belief_maps[1].fill_(0.8)
    bank.timestamp_maps[1].fill_(8.0)
    bank.reset(torch.tensor([True, False]))
    _assert(torch.allclose(bank.belief_maps[0], torch.full_like(bank.belief_maps[0], 0.5)), "Process 0 belief did not reset")
    _assert(torch.allclose(bank.timestamp_maps[0], torch.full_like(bank.timestamp_maps[0], -1.0)), "Process 0 timestamp did not reset")
    _assert(torch.allclose(bank.belief_maps[1], torch.full_like(bank.belief_maps[1], 0.8)), "Process 1 belief was incorrectly reset")
    _assert(torch.allclose(bank.timestamp_maps[1], torch.full_like(bank.timestamp_maps[1], 8.0)), "Process 1 timestamp was incorrectly reset")

    entropy = bank.compute_entropy_maps()
    _assert(bank.belief_maps.shape == (2, 3, 100, 100), "Belief shape mismatch")
    _assert(bank.timestamp_maps.shape == (2, 3, 100, 100), "Timestamp shape mismatch")
    _assert(entropy.shape == (2, 3, 100, 100), "Entropy shape mismatch")
    print("PASS local_map_timestamp_fusion")


def test_decision_inputs_are_comm_limited_and_global_pollution_safe():
    args = _base_args(num_processes=2)
    envs = make_parallel_envs(args)
    try:
        master = setup_master(args)
        obs, env_state, infos = envs.reset()
        polluted_infos = []
        for info in infos:
            info = copy.deepcopy(info)
            info["entropy_map"] = np.full((100, 100), np.nan, dtype=np.float32)
            info["landmark_heatmap"] = np.full((100, 100), np.nan, dtype=np.float32)
            info["voronoi_masks"] = [
                np.full((100, 100), False, dtype=bool)
                for _ in range(args.num_agents)
            ]
            info["map"][1] = np.array([[999.0, 999.0]], dtype=np.float32)
            polluted_infos.append(info)

        master.initialize_obs(obs)
        master.initialize_env_state(env_state)
        master.envs_info = polluted_infos
        with torch.no_grad():
            actions, goals, tasks = master.act(0)

        _assert(np.isfinite(np.asarray(actions)).all(), "Actions became non-finite after global-info pollution")
        _assert(np.isfinite(np.asarray(goals)).all(), "Goals became non-finite after global-info pollution")
        _assert(np.isfinite(np.asarray(tasks)).all(), "Tasks became non-finite after global-info pollution")
        _assert(torch.isfinite(master.local_map_bank.belief_maps).all().item(), "Local belief maps are non-finite")
        _assert(torch.isfinite(master.local_map_bank.compute_entropy_maps()).all().item(), "Local entropy maps are non-finite")

        positions = torch.tensor(
            np.array([info["agent_positions"] for info in polluted_infos]),
            dtype=torch.float32,
        )
        alive = torch.tensor(
            np.array([info["agent_alive_mask"] for info in polluted_infos]),
            dtype=torch.float32,
        )
        comm_mask = build_comm_mask(positions, args.comm_dist, alive)
        rel, low_masks = build_low_level_comm_features(positions, comm_mask)
        _assert(comm_mask.shape == (2, 3, 3), "comm_mask shape mismatch")
        _assert(torch.allclose(rel * (1.0 - low_masks.unsqueeze(-1)), torch.zeros_like(rel)), "Masked low-level rel-pos is not zero")

        for agent in master.all_agents:
            _assert(torch.isfinite(agent.teammate_masks).all().item(), "Stored teammate mask is non-finite")
            _assert(torch.isfinite(agent.explore_nodes).all().item(), "Stored explore nodes are non-finite")
            masked_nodes = agent.teammate_nodes * (1.0 - agent.teammate_masks)
            _assert(torch.allclose(masked_nodes, torch.zeros_like(masked_nodes)), "Masked high-level teammate nodes are not zeroed")
            masked_low_rel = agent.low_teammate_rel_pos * (1.0 - agent.low_teammate_masks.unsqueeze(-1))
            _assert(torch.allclose(masked_low_rel, torch.zeros_like(masked_low_rel)), "Masked low-level teammate rel-pos is not zeroed")
        print("PASS decision_inputs_comm_limited_global_pollution_safe")
    finally:
        envs.close()


def test_landmark_memory_process_reset_and_comm_fusion():
    memory = LandmarkMemory(torch.device("cpu"))
    data = torch.zeros(6, 3, 4)
    mask = torch.zeros(6, 3, 1)
    ts = torch.full((6, 3, 1), -1.0)
    # agent-major linear index: agent_idx * P + proc_idx, P=2.
    data[0, 0, :2] = torch.tensor([0.1, 0.1])
    data[0, 0, 2] = 2.0
    mask[0, 0, 0] = 1.0
    ts[0, 0, 0] = 1.0
    detections = [
        [torch.tensor([[0.2, 0.2]]), torch.zeros(0, 2), torch.zeros(0, 2)],
        [torch.zeros(0, 2), torch.tensor([[0.7, 0.7]]), torch.zeros(0, 2)],
    ]
    comm = torch.eye(3, dtype=torch.bool).view(1, 3, 3).repeat(2, 1, 1)
    comm[0, 1, 0] = True
    out_data, out_mask, out_ts = memory.update(
        data,
        mask,
        ts,
        detections,
        comm,
        torch.tensor([2.0, 2.0]),
        env_dones=torch.tensor([True, False]),
    )
    _assert(out_mask[0].sum().item() == 1.0, "Process 0 local detection after reset missing")
    _assert(torch.allclose(out_data[0, 0, :2], torch.tensor([0.2, 0.2])), "Process 0 reset/update wrong")
    _assert(out_mask[1].sum().item() == 0.0, "Process 1 agent0 should have no detections")
    _assert(out_mask[3].sum().item() == 1.0, "Process 1 agent1 local detection missing")
    _assert(torch.allclose(out_data[3, 0, :2], torch.tensor([0.7, 0.7])), "Process 1 agent1 memory was corrupted")
    print("PASS landmark_memory_process_reset_and_comm_fusion")


def test_rollout_generator_reuses_stored_comm_masks():
    T, P, A = 4, 2, 3
    rollouts = [
        RolloutStorage(T, P, (14,), Discrete(5), A, recurrent_hidden_state_size=8, top_k=3)
        for _ in range(A)
    ]
    for agent_idx, rollout in enumerate(rollouts):
        rollout.low_teammate_masks.copy_(
            torch.arange(T * P * (A - 1), dtype=torch.float32).view(T, P, A - 1)
            + agent_idx * 100.0
        )
        rollout.low_teammate_rel_pos.copy_(
            torch.arange(T * P * (A - 1) * 2, dtype=torch.float32).view(T, P, A - 1, 2)
            + agent_idx * 1000.0
        )
        rollout.goals.copy_(torch.randn_like(rollout.goals))
    advantages = [torch.zeros(T, P, 1) for _ in rollouts]
    sample = next(feed_forward_generator(rollouts, advantages, num_mini_batch=1))
    yielded_rel = sample[-2]
    yielded_masks = sample[-1]
    expected_masks = torch.cat([
        rollout.low_teammate_masks.view(-1, A - 1)
        for rollout in rollouts
    ], dim=0)
    expected_rel = torch.cat([
        rollout.low_teammate_rel_pos.view(-1, A - 1, 2)
        for rollout in rollouts
    ], dim=0)
    _assert(
        torch.equal(torch.sort(yielded_masks.flatten()).values, torch.sort(expected_masks.flatten()).values),
        "PPO generator did not yield stored low teammate masks",
    )
    _assert(
        torch.equal(torch.sort(yielded_rel.flatten()).values, torch.sort(expected_rel.flatten()).values),
        "PPO generator did not yield stored low teammate rel-pos",
    )
    print("PASS rollout_generator_reuses_stored_comm_masks")


def test_low_level_input_shape():
    args = _base_args(num_processes=2)
    master = setup_master(args)
    policy = master.policies_list[0]
    obs = torch.zeros(6, 14)
    goals = torch.zeros(6, 2)
    rel = torch.zeros(6, 2, 2)
    masks = torch.zeros(6, 2)
    low_inp = policy.data_processing_low_level(obs, goals, rel, masks)
    _assert(low_inp.shape == (6, 10), f"Expected 3-agent low input [6,10], got {tuple(low_inp.shape)}")
    _assert(policy.low_level_input == 10, "Policy low_level_input is not 10 for 3 agents")
    print("PASS low_level_input_shape")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    test_local_map_timestamp_fusion()
    test_landmark_memory_process_reset_and_comm_fusion()
    test_rollout_generator_reuses_stored_comm_masks()
    test_low_level_input_shape()
    test_decision_inputs_are_comm_limited_and_global_pollution_safe()
    print("ALL LOCAL COMM VALIDATIONS PASSED")


if __name__ == "__main__":
    sys.exit(main())
