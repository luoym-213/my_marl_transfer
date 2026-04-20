import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time


def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL evaluation: supports eval through training code as well as independently
    policies_list should be a list of policies of all the agents;
    len(policies_list) = num agents
    """
    if env is None or master is None: # if any one of them is None, generate both of them
        master, env = setup_master(args, return_env=True)

    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)
    env.seed(seed)

    if ob_rms is not None:
        obs_mean, obs_std = ob_rms
    else:
        obs_mean = None
        obs_std = None
    master.load_models(policies_list)
    master.set_eval_mode()

    num_eval_episodes = args.num_eval_episodes
    all_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    # TODO: provide support for recurrent policies and mask
    recurrent_hidden_states = None
    mask = None

    # world.dists at the end of episode for simple_spread
    final_min_dists = []
    num_success = 0
    episode_length = 0
    
    # 新增变量，用于记录发现所有landmark的数据
    all_discovery_times = []
    episodes_fully_discovered = 0

    for t in range(num_eval_episodes):
        obs = env.reset()
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        episode_rewards = np.full(env.n, 0.0)
        episode_steps = 0
        
        # 记录landmark发现状态
        num_landmarks = len(env.world.landmarks)
        discovered_landmarks = np.zeros(num_landmarks, dtype=bool)
        episode_discovery_step = None
        
        # 初始检查landmark发现情况
        dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in env.world.landmarks] for a in env.world.agents])
        min_dists_to_landmarks = np.min(dists, axis=0)
        # 用 args.dist_threshold，如果不存在默认为 0.1
        threshold = args.dist_threshold if hasattr(args, 'dist_threshold') else 0.1
        discovered_landmarks |= (min_dists_to_landmarks < threshold)
        if np.all(discovered_landmarks) and episode_discovery_step is None:
            episode_discovery_step = episode_steps

        if render:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            env.render(attn=attn)
            
        while not np.all(done):
            actions = []
            with torch.no_grad():
                actions = master.eval_act(obs, recurrent_hidden_states, mask)
            episode_steps += 1
            obs, reward, done, info = env.step(actions)
            obs = normalize_obs(obs, obs_mean, obs_std)
            episode_rewards += np.array(reward)
            
            # 更新landmark发现情况
            if episode_discovery_step is None:
                dists_step = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in env.world.landmarks] for a in env.world.agents])
                min_dists_step = np.min(dists_step, axis=0)
                discovered_landmarks |= (min_dists_step < threshold)
                if np.all(discovered_landmarks):
                    episode_discovery_step = episode_steps
            
            if render:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                env.render(attn=attn)
                if args.record_video:
                    time.sleep(0.08)

        per_step_rewards[t] = episode_rewards/episode_steps
        num_success += info['n'][0]['is_success']
        episode_length = (episode_length*t + info['n'][0]['world_steps'])/(t+1)

        # 记录每回合全部发现landmark的时间
        if episode_discovery_step is not None:
            all_discovery_times.append(episode_discovery_step)
            episodes_fully_discovered += 1
        else:
            # 如果直到回合结束都没有全部发现，可以用最大步数作为时间，或者选择忽略
            all_discovery_times.append(episode_steps)

        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.world.min_dists)
        elif args.env_name == 'simple_formation' or args.env_name=='simple_line':
            final_min_dists.append(env.world.dists)

        if render:
            print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['n'][0]['is_success'],
                per_step_rewards[t][0],info['n'][0]['world_steps']))
        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents

        if args.record_video:
            # print(attn)
            input('Press enter to continue: ')
    
    # 计算全部发现landmark的平均时间
    average_discovery_time = np.mean(all_discovery_times) if all_discovery_times else 0
                
    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length, average_discovery_time, episodes_fully_discovered


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length, average_discovery_time, episodes_fully_discovered = evaluate(args, args.seed, 
                    policies_list, ob_rms, args.render, render_attn=args.masking)
    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f})"
            .format(per_step_rewards.mean(0),num_success,args.num_eval_episodes,episode_length))
    print("Average Time to Discover All Landmarks: {:.2f} steps (Fully Discovered in {}/{} episodes)"
            .format(average_discovery_time, episodes_fully_discovered, args.num_eval_episodes))
    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))
