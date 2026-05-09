import datetime
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

from marl.agents.learner import setup_master
from marl.controllers.random_goals import RandomGoalManager
from marl import utils


class LowLevelMetrics:
    def __init__(self, boundary, collision_distance=0.15):
        self.boundary = boundary
        self.collision_distance = collision_distance
        self.reset()

    def reset(self):
        self.reward_sum = 0.0
        self.reward_count = 0
        self.distance_sum = 0.0
        self.distance_count = 0
        self.collision_sum = 0.0
        self.boundary_sum = 0.0
        self.event_count = 0

    def update(self, reward, envs_info, goals):
        reward = np.asarray(reward, dtype=np.float32)
        self.reward_sum += float(reward.sum())
        self.reward_count += reward.size

        positions = np.asarray(
            [info["agent_positions"] for info in envs_info],
            dtype=np.float32,
        )
        distances = np.linalg.norm(positions - goals, axis=-1)
        self.distance_sum += float(distances.sum())
        self.distance_count += distances.size

        collisions = self._collision_indicators(positions)
        boundary = np.any(np.abs(positions) >= self.boundary, axis=-1)
        self.collision_sum += float(collisions.sum())
        self.boundary_sum += float(boundary.sum())
        self.event_count += positions.shape[0] * positions.shape[1]

    def snapshot(self):
        return {
            "average_low_level_reward": self.reward_sum / max(1, self.reward_count),
            "average_distance_to_goal": self.distance_sum / max(1, self.distance_count),
            "collision_rate": self.collision_sum / max(1, self.event_count),
            "boundary_violation_rate": self.boundary_sum / max(1, self.event_count),
        }

    def _collision_indicators(self, positions):
        num_processes, num_agents, _ = positions.shape
        indicators = np.zeros((num_processes, num_agents), dtype=np.float32)
        for proc_idx in range(num_processes):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(positions[proc_idx, i] - positions[proc_idx, j])
                    if dist < self.collision_distance:
                        indicators[proc_idx, i] = 1.0
                        indicators[proc_idx, j] = 1.0
        return indicators


class LowLevelTrainer:
    """Training loop for the first stage: low-level goal-conditioned policy."""

    def __init__(self, args):
        self.args = args
        self.writer = None
        self.envs = None
        self.master = None
        self.num_agents = None
        self.goal_manager = None
        self.metrics = LowLevelMetrics(boundary=float(args.arena_size))
        self.start_time = None
        self.savedir = None
        self.return_early = False

    def setup(self):
        args = self.args
        self.writer = SummaryWriter(args.log_dir)
        self.envs = utils.make_parallel_envs(args)
        self.master = setup_master(args)

        obs, env_state, reset_info = self.envs.reset()
        print("obs shape: ", obs.shape)
        print("state shape: ", env_state.shape)

        self.master.initialize_obs(obs)
        self.master.initialize_env_state(env_state)
        self.master.envs_info = reset_info
        self.num_agents = len(self.master.all_agents)

        self.goal_manager = RandomGoalManager(
            args.num_processes,
            self.num_agents,
            float(args.arena_size),
            rng=np.random.RandomState(args.seed),
        )
        self.goal_manager.initialize(env_state=env_state, envs_info=reset_info)
        self.start_time = datetime.datetime.now()

    def train(self, return_early=False):
        self.return_early = return_early
        self.setup()

        try:
            for update_idx in range(self.args.num_updates):
                self.collect_rollout()
                losses = self.update()

                if update_idx % self.args.save_interval == 0 and not self.args.test:
                    self.save_checkpoint(update_idx)

                if update_idx % self.args.log_interval == 0:
                    self.log_training(update_idx, losses)
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.envs is not None:
                self.envs.close()

        if return_early:
            return self.savedir
        return None

    def collect_rollout(self):
        args = self.args

        for step in range(args.num_steps):
            goals_for_step = self.goal_manager.goals.copy()
            with torch.no_grad():
                actions_list, goals_list, tasks_list = self.master.act_low_level(
                    step,
                    goals_for_step,
                )

            agent_actions = np.transpose(np.array(actions_list), (1, 0, 2))
            agent_goals = np.transpose(np.array(goals_list), (1, 0, 2))
            agent_tasks = np.transpose(np.array(tasks_list), (1, 0, 2))

            step_data = [
                {
                    "agents_actions": agent_actions[i],
                    "agents_goals": agent_goals[i],
                    "agents_tasks": agent_tasks[i],
                }
                for i in range(args.num_processes)
            ]

            obs, reward, high_reward, done_info, info, env_state = self.envs.step(step_data)
            del high_reward
            done = np.array([done_info[i]["all"] for i in range(args.num_processes)])
            done_agent = np.array([done_info[i]["agent"] for i in range(args.num_processes)])

            self.master.envs_info = info
            self.metrics.update(reward, info, goals_for_step)

            reward_t = torch.from_numpy(np.stack(reward)).float().to(args.device)
            high_reward_t = torch.zeros_like(reward_t)
            all_masks = torch.FloatTensor(1 - 1.0 * done).to(args.device)
            masks = torch.FloatTensor(1 - 1.0 * done_agent).to(args.device)
            goal_dones = torch.FloatTensor(
                [info[i]["goal_done"] for i in range(args.num_processes)]
            ).to(args.device)

            self.master.update_rollout(
                obs,
                reward_t,
                high_reward_t,
                masks,
                env_state,
                goal_dones,
            )
            self.goal_manager.advance_and_resample(
                info,
                done_agent,
                env_state=env_state,
            )

    def update(self):
        self.master.wrap_low_horizon(self.goal_manager.goals)
        return_vals = self.master.update_low_level()
        self.master.after_update()
        return {
            "value_low_loss": return_vals[:, 0],
            "action_low_loss": return_vals[:, 1],
            "dist_low_entropy": return_vals[:, 2],
        }

    def save_checkpoint(self, update_idx):
        if not self.master.trainers_list:
            return

        module_dir = os.path.join(
            self.args.save_dir,
            "comm_low_policy",
            f"ep{update_idx}_modules",
        )
        os.makedirs(module_dir, exist_ok=True)
        self.savedir = os.path.join(module_dir, "low_level.pth")

        policy = self.master.policies_list[0]
        trainer = self.master.trainers_list[0]
        checkpoint = {
            "module_name": "low_level",
            "state_dict": policy.modules_dict["low_level"].state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": {
                "num_agents": policy.num_agents,
                "num_entities": policy.num_entities,
                "hidden_dim": policy.h_dim,
                "embed_dim": policy.embed_dim,
                "goal_dim": 2,
            },
        }
        torch.save(checkpoint, self.savedir)
        print(f"Saved low-level checkpoint to {self.savedir}")

    def log_training(self, update_idx, losses):
        args = self.args
        end = datetime.datetime.now()
        seconds = max((end - self.start_time).total_seconds(), 1e-6)
        total_num_steps = (update_idx + 1) * args.num_processes * args.num_steps

        metric_values = self.metrics.snapshot()
        metric_values.update(self.goal_manager.metrics_snapshot())
        value_low_loss = losses["value_low_loss"]
        action_low_loss = losses["action_low_loss"]
        dist_low_entropy = losses["dist_low_entropy"]

        print(
            "Updates {} | Num timesteps {} | Time {} | FPS {} "
            "\nAverage low-level reward {:.4f} | Goal success rate {:.4f} | "
            "Average distance to goal {:.4f}"
            "\nCollision rate {:.4f} | Boundary violation rate {:.4f} | "
            "Average steps per goal {:.2f}"
            "\nlow Entropy {:.4f} low Value loss {:.4f} low level loss {:.4f}\n".format(
                update_idx,
                total_num_steps,
                str(end - self.start_time),
                int(total_num_steps / seconds),
                metric_values["average_low_level_reward"],
                metric_values["goal_success_rate"],
                metric_values["average_distance_to_goal"],
                metric_values["collision_rate"],
                metric_values["boundary_violation_rate"],
                metric_values["average_steps_per_goal"],
                dist_low_entropy[0],
                value_low_loss[0],
                action_low_loss[0],
            )
        )

        if args.test:
            self.metrics.reset()
            return

        self.writer.add_scalar(
            "low/average_reward",
            metric_values["average_low_level_reward"],
            update_idx,
        )
        self.writer.add_scalar(
            "low/goal_success_rate",
            metric_values["goal_success_rate"],
            update_idx,
        )
        self.writer.add_scalar(
            "low/average_distance_to_goal",
            metric_values["average_distance_to_goal"],
            update_idx,
        )
        self.writer.add_scalar(
            "low/collision_rate",
            metric_values["collision_rate"],
            update_idx,
        )
        self.writer.add_scalar(
            "low/boundary_violation_rate",
            metric_values["boundary_violation_rate"],
            update_idx,
        )
        self.writer.add_scalar(
            "low/average_steps_per_goal",
            metric_values["average_steps_per_goal"],
            update_idx,
        )
        self.writer.add_scalar("low/value_loss", value_low_loss[0], update_idx)
        self.writer.add_scalar("low/action_loss", action_low_loss[0], update_idx)
        self.writer.add_scalar("low/dist_entropy", dist_low_entropy[0], update_idx)
        self.metrics.reset()
