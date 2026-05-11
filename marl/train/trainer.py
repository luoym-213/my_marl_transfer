import datetime
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval import evaluate as evaluate_policy
from marl.agents.learner import setup_master
from marl import utils
from marl.timing import TimingProfiler


class Trainer:
    """Owns the end-to-end training loop.

    This is a direct extraction of the previous main.train function. The first
    refactor step keeps behavior and data flow unchanged while moving training
    responsibilities out of main.py.
    """

    def __init__(self, args):
        self.args = args
        self.writer = None
        self.envs = None
        self.master = None
        self.eval_master = None
        self.eval_env = None

        self.num_agents = None
        self.episode_rewards = None
        self.final_rewards = None
        self.episode_high_rewards = None
        self.final_high_rewards = None

        self.start_time = None
        self.savedir = None
        self.return_early = False
        self.timer = TimingProfiler(
            enabled=getattr(args, "timing_profile", False),
            cuda=getattr(args, "cuda", False),
            log_interval=getattr(args, "timing_log_interval", 10),
        )

    def setup(self):
        args = self.args

        self.writer = SummaryWriter(args.log_dir)
        args.timing_timer = self.timer
        self.envs = utils.make_parallel_envs(args)
        self.master = setup_master(args)
        self.eval_master, self.eval_env = setup_master(args, return_env=True)

        obs, env_state, reset_info = self.envs.reset()

        print("obs shape: ", obs.shape)
        print("state shape: ", env_state.shape)

        self.master.initialize_obs(obs)
        self.master.initialize_env_state(env_state)
        self.master.envs_info = reset_info

        self.num_agents = len(self.master.all_agents)
        self.episode_rewards = torch.zeros(
            [args.num_processes, self.num_agents], device=args.device
        )
        self.final_rewards = torch.zeros(
            [args.num_processes, self.num_agents], device=args.device
        )
        self.episode_high_rewards = torch.zeros(
            [args.num_processes, self.num_agents], device=args.device
        )
        self.final_high_rewards = torch.zeros(
            [args.num_processes, self.num_agents], device=args.device
        )

        self.start_time = datetime.datetime.now()

    def train(self, return_early=False):
        self.return_early = return_early
        self.setup()

        try:
            for update_idx in range(self.args.num_updates):
                self.timer.reset()
                self.collect_rollout()
                losses = self.update()

                if update_idx % self.args.save_interval == 0 and not self.args.test:
                    with self.timer.time("save_checkpoint"):
                        self.save_checkpoint(update_idx)

                if update_idx % self.args.log_interval == 0:
                    self.log_training(update_idx, losses)

                should_stop = False
                if self.should_evaluate(update_idx):
                    with self.timer.time("evaluation_total"):
                        should_stop = self.evaluate(update_idx)

                self.timer.finish_update(update_idx, writer=None if self.args.test else self.writer)
                if should_stop:
                    break
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.envs is not None:
                self.envs.close()
            if self.eval_env is not None and hasattr(self.eval_env, "close"):
                self.eval_env.close()

        if return_early:
            return self.savedir
        return None

    def collect_rollout(self):
        """
        Collects a rollout of experience by acting in the environment for num_steps steps, then stores the experience in the master.
        """

        args = self.args

        with self.timer.time("rollout_total"):
            for step in range(args.num_steps):
                with torch.no_grad():
                    with self.timer.time("master_act_total"):
                        actions_list, goals_list, tasks_list = self.master.act(step)

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

                with self.timer.time("env_step", sync_cuda=False):
                    obs, reward, high_reward, done_info, info, env_state = self.envs.step(step_data)
                done = np.array([done_info[i]["all"] for i in range(args.num_processes)])
                done_agent = np.array([done_info[i]["agent"] for i in range(args.num_processes)])

                self.master.envs_info = info

                high_reward = torch.from_numpy(np.stack(high_reward)).float().to(args.device)
                reward = torch.from_numpy(np.stack(reward)).float().to(args.device)

                self.episode_rewards += reward
                self.episode_high_rewards += high_reward

                all_masks = torch.FloatTensor(1 - 1.0 * done).to(args.device)
                masks = torch.FloatTensor(1 - 1.0 * done_agent).to(args.device)
                goal_dones = torch.FloatTensor(
                    [info[i]["goal_done"] for i in range(args.num_processes)]
                ).to(args.device)

                self.final_rewards *= all_masks
                self.final_rewards += (1 - all_masks) * self.episode_rewards
                self.final_high_rewards *= all_masks
                self.final_high_rewards += (1 - all_masks) * self.episode_high_rewards

                self.episode_rewards *= all_masks
                self.episode_high_rewards *= all_masks

                with self.timer.time("rollout_write/update_rollout"):
                    self.master.update_rollout(
                        obs, reward, high_reward, masks, env_state, goal_dones
                    )

    def update(self):
        with self.timer.time("wrap_horizon"):
            self.master.wrap_horizon()
        with self.timer.time("ppo_update_total"):
            return_vals = self.master.update()
        self.master.after_update()

        return {
            "value_low_loss": return_vals[:, 0],
            "action_low_loss": return_vals[:, 1],
            "dist_low_entropy": return_vals[:, 2],
            "value_high_loss": return_vals[:, 3],
            "goal_high_loss": return_vals[:, 4],
            "goal_entropy": return_vals[:, 5],
        }

    def save_checkpoint(self, update_idx, save_modules=True):
        args = self.args
        savedict = {
            "models": [
                agent.actor_critic.state_dict() for agent in self.master.all_agents
            ]
        }
        savedict["ob_rms"] = self.get_ob_rms()

        self.savedir = os.path.join(args.save_dir, "ep" + str(update_idx) + ".pt")
        torch.save(savedict, self.savedir)

        if save_modules and len(self.master.all_agents) > 0:
            module_dir = os.path.join(args.save_dir, f"ep{update_idx}_modules")
            os.makedirs(module_dir, exist_ok=True)
            self.master.all_agents[0].actor_critic.save_all_modules(module_dir)
            print(f"Saved shared modular checkpoints to {module_dir}")

    def log_training(self, update_idx, losses):
        args = self.args
        end = datetime.datetime.now()
        seconds = (end - self.start_time).total_seconds()
        total_num_steps = (update_idx + 1) * args.num_processes * args.num_steps

        mean_low_reward = self.final_rewards.mean(dim=0).cpu().numpy()
        mean_high_reward = self.final_high_rewards.mean(dim=0).cpu().numpy()

        value_low_loss = losses["value_low_loss"]
        action_low_loss = losses["action_low_loss"]
        dist_low_entropy = losses["dist_low_entropy"]
        value_high_loss = losses["value_high_loss"]
        goal_high_loss = losses["goal_high_loss"]
        goal_entropy = losses["goal_entropy"]

        print(
            "Updates {} | Num timesteps {} | Time {} | FPS {} "
            "\nMean low reward {} low Entropy {:.4f} low Value loss {:.4f} low level loss {:.4f} "
            "\nMean high reward {} high Entropy {:.4f}  high Value loss {:.4f} high level loss {:.4f}\n ".format(
                update_idx,
                total_num_steps,
                str(end - self.start_time),
                int(total_num_steps / seconds),
                mean_low_reward,
                dist_low_entropy[0],
                value_low_loss[0],
                action_low_loss[0],
                mean_high_reward,
                goal_entropy[0],
                value_high_loss[0],
                goal_high_loss[0],
            )
        )

        if args.test:
            return

        for idx in range(self.num_agents):
            self.writer.add_scalar(
                "agent" + str(idx) + "/training_low_reward",
                mean_low_reward[idx],
                update_idx,
            )
            self.writer.add_scalar(
                "agent" + str(idx) + "/training_high_reward",
                mean_high_reward[idx],
                update_idx,
            )

        self.writer.add_scalar("all/low_value_loss", value_low_loss[0], update_idx)
        self.writer.add_scalar("all/action_low_loss", action_low_loss[0], update_idx)
        self.writer.add_scalar("all/dist_low_entropy", dist_low_entropy[0], update_idx)
        self.writer.add_scalar("all/high_value_loss", value_high_loss[0], update_idx)
        self.writer.add_scalar("all/goal_high_loss", goal_high_loss[0], update_idx)
        self.writer.add_scalar("all/goal_entropy", goal_entropy[0], update_idx)

    def should_evaluate(self, update_idx):
        return (
            self.args.eval_interval is not None
            and update_idx % self.args.eval_interval == 0
        )

    def evaluate(self, update_idx):
        args = self.args
        print(
            "==========================================================================================="
        )
        (
            _,
            eval_perstep_rewards,
            _,
            eval_high_perstep_rewards,
            final_min_dists,
            num_success,
            eval_episode_len,
            _,
            _,
        ) = evaluate_policy(
            args,
            None,
            self.master.all_policies,
            ob_rms=self.get_ob_rms(),
            env=self.eval_env,
            master=self.eval_master,
            render=args.render,
        )
        print(
            "Evaluation {:d} | Mean per-step reward {:.2f}".format(
                update_idx // args.eval_interval, eval_perstep_rewards.mean()
            )
        )
        print(
            "Mean high-level per-step reward {:.2f}".format(
                eval_high_perstep_rewards.mean()
            )
        )
        print(
            "Num success {:d}/{:d} | Episode Length {:.2f}".format(
                num_success, args.num_eval_episodes, eval_episode_len
            )
        )
        if final_min_dists:
            print("Final_dists_mean {}".format(np.stack(final_min_dists).mean(0)))
            print("Final_dists_var {}".format(np.stack(final_min_dists).var(0)))
        print(
            "===========================================================================================\n"
        )

        if not args.test:
            self.log_evaluation(
                update_idx,
                eval_perstep_rewards,
                eval_high_perstep_rewards,
                final_min_dists,
                num_success,
                eval_episode_len,
            )

        curriculum_success_thres = 0.9
        if (
            self.return_early
            and num_success * 1.0 / args.num_eval_episodes > curriculum_success_thres
        ):
            self.save_checkpoint(update_idx, save_modules=False)
            print(
                "===========================================================================================\n"
            )
            print("{} agents: training complete. Breaking.\n".format(args.num_agents))
            print(
                "===========================================================================================\n"
            )
            return True

        return False

    def log_evaluation(
        self,
        update_idx,
        eval_perstep_rewards,
        eval_high_perstep_rewards,
        final_min_dists,
        num_success,
        eval_episode_len,
    ):
        args = self.args
        self.writer.add_scalar(
            "all/eval_success",
            100.0 * num_success / args.num_eval_episodes,
            update_idx,
        )
        self.writer.add_scalar("all/episode_length", eval_episode_len, update_idx)

        for idx in range(self.num_agents):
            self.writer.add_scalar(
                "agent" + str(idx) + "/eval_per_step_reward",
                eval_perstep_rewards.mean(0)[idx],
                update_idx,
            )
            self.writer.add_scalar(
                "agent" + str(idx) + "/eval_high_per_step_reward",
                eval_high_perstep_rewards.mean(0)[idx],
                update_idx,
            )
            if final_min_dists:
                self.writer.add_scalar(
                    "agent" + str(idx) + "/eval_min_dist",
                    np.stack(final_min_dists).mean(0)[idx],
                    update_idx,
                )

    def get_ob_rms(self):
        if self.envs.ob_rms is None:
            return None, None
        return self.envs.ob_rms[0].mean, self.envs.ob_rms[0].var
