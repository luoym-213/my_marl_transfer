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
    master = setup_master(args) # setup learner ，no env
    # used during evaluation only
    eval_master, eval_env = setup_master(args, return_env=True)  # setup evaluate learner with SINGLE env
    obs, env_state, reset_info = envs.reset() # shape - num_processes x num_agents x obs_dim

    print("obs shape: ", obs.shape)
    print("state shape: ", env_state.shape)
    master.initialize_obs(obs)
    master.initialize_env_state(env_state)
    master.envs_info = reset_info

    n = len(master.all_agents)
    episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_rewards = torch.zeros([args.num_processes, n], device=args.device)
    episode_high_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_high_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    for j in range(args.num_updates):
        for step in range(args.num_steps):
            with torch.no_grad():
                print("step: ", step)
                actions_list, goals_list = master.act(step)
            agent_actions = np.transpose(np.array(actions_list),(1,0,2))
            agent_goals = np.transpose(np.array(goals_list),(1,0,2))
            # 这里需要给agent_actions和agent_goals包装一下，变成字典形式传入env.step()
            # 需要变成num_processes个环境的列表，每个元素是一个字典，包含两个键值对: 'agents_actions', 'agents_goals'
            step_data = [{'agents_actions': agent_actions[i], 'agents_goals': agent_goals[i]} for i in range(args.num_processes)]
            obs, reward, high_reward, done, info, env_state = envs.step(step_data)
            master.envs_info = info
            high_reward = torch.from_numpy(np.stack(high_reward)).float().to(args.device)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device)
            episode_rewards += reward
            episode_high_rewards += high_reward
            masks = torch.FloatTensor(1-1.0*done).to(args.device)
            goal_dones = torch.FloatTensor([info[i]['goal_done'] for i in range(args.num_processes)]).to(args.device)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            final_high_rewards *= masks
            final_high_rewards += (1 - masks) * episode_high_rewards
            episode_rewards *= masks
            episode_high_rewards *= masks

            master.update_rollout(obs, reward, high_reward, masks, env_state, goal_dones)

        master.wrap_horizon()
        return_vals = master.update()
        value_low_loss = return_vals[:, 0]
        action_low_loss = return_vals[:, 1]
        dist_low_entropy = return_vals[:, 2]
        value_high_loss = return_vals[:, 3]
        decision_high_loss = return_vals[:, 4]
        waypoint_high_loss = return_vals[:, 5]
        decision_entropy = return_vals[:, 6]
        waypoint_entropy = return_vals[:, 7]
        master.after_update()

        if j%args.save_interval == 0 and not args.test:
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir+'/ep'+str(j)+'.pt'
            torch.save(savedict, savedir)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j%args.log_interval == 0:
            end = datetime.datetime.now()
            seconds = (end-start).total_seconds()
            mean_low_reward = final_rewards.mean(dim=0).cpu().numpy()
            mean_high_reward = final_high_rewards.mean(dim=0).cpu().numpy()
            print("Updates {} | Num timesteps {} | Time {} | FPS {} \
                  \nMean low reward {} low Entropy {:.4f} low Value loss {:.4f} lowPolicy loss {:.4f} \
                  \nMean high reward {} Decision Entropy {:.4f} Waypoint Entropy {:.4f} high Value loss {:.4f} high Decision loss {:.4f} high Waypoint loss {:.4f}\n "
            .format(j, total_num_steps, str(end-start), int(total_num_steps / seconds), 
                  mean_low_reward, dist_low_entropy[0], value_low_loss[0], action_low_loss[0],
                  mean_high_reward, decision_entropy[0], waypoint_entropy[0], value_high_loss[0],
                  decision_high_loss[0], waypoint_high_loss[0]))
            
            if not args.test:
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/training_low_reward', mean_low_reward[idx], j)
                    writer.add_scalar('agent'+str(idx)+'/training_high_reward', mean_high_reward[idx], j)

                writer.add_scalar('all/low_value_loss', value_low_loss[0], j)
                writer.add_scalar('all/action_low_loss', action_low_loss[0], j)
                writer.add_scalar('all/dist_low_entropy', dist_low_entropy[0], j)
                writer.add_scalar('all/high_value_loss', value_high_loss[0], j)
                writer.add_scalar('all/decision_high_loss', decision_high_loss[0], j)
                writer.add_scalar('all/waypoint_high_loss', waypoint_high_loss[0], j)
                writer.add_scalar('all/decision_entropy', decision_entropy[0], j)
                writer.add_scalar('all/waypoint_entropy', waypoint_entropy[0], j)

        if args.eval_interval is not None and j%args.eval_interval==0:
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            print('===========================================================================================')
            _, eval_perstep_rewards, _, eval_high_perstep_rewards, final_min_dists, num_success, eval_episode_len, _, _ = evaluate(args, None, master.all_policies,
                                                                                               ob_rms=ob_rms, env=eval_env,
                                                                                               master=eval_master, render=args.render)
            print('Evaluation {:d} | Mean per-step reward {:.2f}'.format(j//args.eval_interval, eval_perstep_rewards.mean()))
            print('Mean high-level per-step reward {:.2f}'.format(eval_high_perstep_rewards.mean()))
            print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
            if final_min_dists:
                print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
                print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
            print('===========================================================================================\n')

            if not args.test:
                writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, j)
                writer.add_scalar('all/episode_length', eval_episode_len, j)
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], j)
                    writer.add_scalar('agent'+str(idx)+'/eval_high_per_step_reward', eval_high_perstep_rewards.mean(0)[idx], j)
                    if final_min_dists:
                        writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], j)

            curriculum_success_thres = 0.9
            if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
                savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
                ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                savedict['ob_rms'] = ob_rms
                savedir = args.save_dir+'/ep'+str(j)+'.pt'
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
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
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
