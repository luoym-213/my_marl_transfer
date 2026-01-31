import os
import json
import datetime
import numpy as np
import torch
import utils
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter
from eval import evaluate
from learner import setup_master
from pprint import pprint

np.set_printoptions(suppress=True, precision=4)

def train(args, return_early=False):
    writer = SummaryWriter(args.log_dir)    
    envs = utils.make_parallel_envs(args) # make parallel envs
    master = setup_master(args) # setup learner
    # used during evaluation only
    eval_master, eval_env = setup_master(args, return_env=True)  # setup evaluate learner with SINGLE env

    n = len(master.all_agents)
    eta = 0.1  # weight for action smooth loss
    episode_rewards = np.zeros([args.num_processes, n])
    final_rewards = np.zeros([args.num_processes, n])
    episode_high_rewards = np.zeros([args.num_processes, n])
    final_high_rewards = np.zeros([args.num_processes, n])

    # start simulations
    start = datetime.datetime.now()
    for t in range(args.total_steps):
        # for each episode, initia
        if t % args.episode_steps == 0:
            obs, env_state, reset_info = envs.reset() 
            # obs: [num_processes, num_agents, obs_dim]
            # env_state: [num_processes,  global_state_dim（agent * 6）]

            # allocate region for each agent and compute initial reward
            belief_maps = np.array([reset_info[i]['belief_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]
            visited_maps = np.array([reset_info[i]['visited_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]
            curr_map_entropy = np.array([reset_info[i]['entropy_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]
            
            # execute high-level action
            with torch.no_grad():
                R_u, g_u = master.high_level_act(obs, belief_maps)      # [num_processes, num_agents, goal_dim]
            r_init = master.compute_initial_reward(R_u, obs, visited_maps)  # [num_processes, num_agents]

            last_goals = g_u
            last_obs = obs
            last_belief_maps = belief_maps
            last_visited_maps = visited_maps
            last_map_entropy = curr_map_entropy

            value_high_loss = np.zeros((args.num_agents,1))
            goal_high_loss = np.zeros((args.num_agents,1))

        # === High-Level Decision (every K steps) ===
        if t % args.K == 0 and t % args.episode_steps != 0:
            # 0. save last high level decision data (t-K)
            next_belief_maps = belief_maps
            next_obs = obs
            next_visited_maps = visited_maps
            
            prev_goals = last_goals
            last_goals = g_u

            # 1. get current state (t)
            curr_map_entropy = np.array([info[i]['entropy_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]
            
            # 2. compute high-level reward (t-K to t)
            # a. compute each agent's ref reward from (t-K) to (t)
            r_ref = master.compute_r_ref(R_u, last_map_entropy, curr_map_entropy)   # [num_processes, num_agents]

            # b. action smooth loss
            smooth_loss = np.linalg.norm(last_goals - prev_goals, axis=2)  # [num_processes, num_agents]

            # d. total high-level reward
            beta = master.get_beta(t, args.episode_steps)
            high_level_rewards = (1-beta) * r_init + beta * r_ref - eta * smooth_loss   # [num_processes, num_agents]

            # 3. add to buffer...
            master.update_high_buffer(last_belief_maps, last_obs, R_u, last_goals,
                                    high_level_rewards, next_belief_maps, next_obs, done)
            
            # 3.5 accumulate episode high-level reward
            episode_high_rewards += high_level_rewards
            final_high_rewards *= all_masks
            final_high_rewards += (1 - all_masks) * episode_high_rewards
            episode_high_rewards *= all_masks

            # 4. execute new high-level action (t)
            R_u, g_u = master.high_level_act(obs, belief_maps)
            r_init = master.compute_initial_reward(R_u, next_obs, next_visited_maps)

            # 5. save for next high-level reward computation
            last_belief_maps = belief_maps
            last_obs = next_obs
            last_visited_maps = next_visited_maps
            last_map_entropy = curr_map_entropy

            # 6. update high-level net
            return_high_vals = master.update_high_level()
            return_high_vals = np.array(return_high_vals)
            value_high_loss = return_high_vals[:, 0]
            goal_high_loss = return_high_vals[:, 1]

        # === Low-Level Decision (every step) ===
        agent_actions, action_int = master.low_level_act(obs, g_u)
        step_data = [{'agents_actions': action_int[i]} for i in range(args.num_processes)]
        next_obs, rew, low_level_rewards, done_info, info, env_state = envs.step(step_data)
        next_belief_maps = np.array([info[i]['belief_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]

        # 1. add to buffer...
        done = np.array([done_info[i]['all'] for i in range(args.num_processes)])
        master.update_low_buffer(obs, belief_maps, g_u, agent_actions, low_level_rewards[:, :, np.newaxis], next_obs, next_belief_maps, done[:, :, np.newaxis])

        # 1.5 compute low-level reward
        episode_rewards += low_level_rewards
        all_masks = 1.0 - done.astype(np.float32)
        final_rewards *= all_masks
        final_rewards += (1 - all_masks) * episode_rewards
        final_high_rewards *= all_masks
        final_high_rewards += (1 - all_masks) * episode_high_rewards
        episode_rewards *= all_masks
        episode_high_rewards *= all_masks

        # 2. save for next step
        obs = next_obs
        belief_maps = next_belief_maps
        visited_maps = np.array([info[i]['visited_map'] for i in range(args.num_processes)])  # [num_processes, map_dim, map_dim]

        # 3. update low-level net
        return_low_vals = master.update_low_level()
        return_low_vals = np.array(return_low_vals)
        value_low_loss = return_low_vals[:, 0]
        action_low_loss = return_low_vals[:, 1]

        if t%args.save_interval == 0 and not args.test:
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir+'/ep'+str(t)+'.pt'
            torch.save(savedict, savedir)

            # 新增：按模块拆分保存（每个 agent 独立目录）
            # 修改后：只保存第一个智能体的模块参数（因为参数共享）
            if len(master.all_agents) > 0:
                # 创建保存目录，不再区分 agent0, agent1...
                module_dir = os.path.join(args.save_dir, f'ep{t}_modules')
                os.makedirs(module_dir, exist_ok=True)
                
                # 只调用第一个智能体的保存函数
                master.all_agents[0].actor_critic.save_all_modules(module_dir)
                print(f"✅ Saved shared modular checkpoints to {module_dir}")

        total_num_steps = (t + 1) * args.num_processes * args.num_steps

        if t%args.log_interval == 0:
            end = datetime.datetime.now()
            seconds = (end-start).total_seconds()
            mean_low_reward = final_rewards.mean(axis=0)
            mean_high_reward = final_high_rewards.mean(axis=0)

            print("Updates {} | Num timesteps {} | Time {} | FPS {} \
                  \nlow Value loss {:.4f} low level loss {:.4f} \
                  \nhigh Value loss {:.4f} high level loss {:.4f}\n "
            .format(t, total_num_steps, str(end-start), int(total_num_steps / seconds), 
                  float(value_low_loss[0]), float(action_low_loss[0]),
                  float(value_high_loss[0]), float(goal_high_loss[0])))
            
            if not args.test:
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/training_low_reward', float(mean_low_reward[idx]), t)
                    writer.add_scalar('agent'+str(idx)+'/training_high_reward', float(mean_high_reward[idx]), t)

                writer.add_scalar('all/low_value_loss', value_low_loss[0], t)
                writer.add_scalar('all/action_low_loss', action_low_loss[0], t)
                writer.add_scalar('all/high_value_loss', value_high_loss[0], t)
                writer.add_scalar('all/goal_high_loss', goal_high_loss[0], t)

        if args.eval_interval is not None and t%args.eval_interval==0:
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            print('===========================================================================================')
            _, eval_perstep_rewards, _, eval_high_perstep_rewards, final_min_dists, num_success, eval_episode_len, _, _ = evaluate(args, None, master.all_policies,
                                                                                               ob_rms=ob_rms, env=eval_env,
                                                                                               master=eval_master, render=args.render)
            print('Evaluation {:d} | Mean per-step reward {:.2f}'.format(t//args.eval_interval, eval_perstep_rewards.mean()))
            print('Mean high-level per-step reward {:.2f}'.format(eval_high_perstep_rewards.mean()))
            print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
            if final_min_dists:
                print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
                print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
            print('===========================================================================================\n')

            if not args.test:
                writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, t)
                writer.add_scalar('all/episode_length', eval_episode_len, t)
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], t)
                    writer.add_scalar('agent'+str(idx)+'/eval_high_per_step_reward', eval_high_perstep_rewards.mean(0)[idx], t)
                    if final_min_dists:
                        writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], t)

            curriculum_success_thres = 0.9
            if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
                savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
                ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                savedict['ob_rms'] = ob_rms
                savedir = args.save_dir+'/ep'+str(t)+'.pt'
                torch.save(savedict, savedir)
                print('===========================================================================================\n')
                print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
                print('===========================================================================================\n')
                break

    writer.close()
    if return_early:
        return savedir

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.total_steps = args.num_frames // args.num_processes
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    train(args)
