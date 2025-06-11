import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time
import os
import imageio

# Global counter to track evaluate function calls
_evaluate_call_count = 0

def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL evaluation: supports eval through training code as well as independently
    policies_list should be a list of policies of all the agents;
    len(policies_list) = num agents
    """
    global _evaluate_call_count
    _evaluate_call_count += 1
    current_eval_call = _evaluate_call_count
    
    if env is None or master is None: # if any one of them is None, generate both of them
        master, env = setup_master(args, return_env=True)

    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)
    print(f"Evaluation Call #{current_eval_call}")
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

    # Create evaluation-specific folder if record_video is enabled
    eval_folder = None
    if args.record_video:
        eval_folder = os.path.join(args.gif_save_path, f"evaluation_{current_eval_call}")
        os.makedirs(eval_folder, exist_ok=True)
        print(f"GIF files will be saved to: {eval_folder}")

    for t in range(num_eval_episodes):
        obs = env.reset()
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        episode_rewards = np.full(env.n, 0.0)
        episode_steps = 0
        
        # Determine rendering behavior
        # If record_video is True, we always want to save GIF for first 5 episodes
        # If render is True, we show the window for first 5 episodes
        should_save_gif = args.record_video and t < 5
        should_show_window = render and t < 5
        
        # Initialize frame collection for GIF
        frames = []
        
        # Initial render for GIF saving (if needed)
        if should_save_gif:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            render_result = env.render(mode='rgb_array', attn=attn)
            if render_result:
                frames.append(render_result[0])
        
        # Show window if render is enabled
        if should_show_window:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            env.render(mode='human', attn=attn)
            
        while not np.all(done):
            actions = []
            with torch.no_grad():
                actions = master.eval_act(obs, recurrent_hidden_states, mask)
            episode_steps += 1
            obs, reward, done, info = env.step(actions)
            obs = normalize_obs(obs, obs_mean, obs_std)
            episode_rewards += np.array(reward)
            
            # Render for GIF saving (if needed)
            if should_save_gif:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                render_result = env.render(mode='rgb_array', attn=attn)
                if render_result:
                    frames.append(render_result[0])
            
            # Show window if render is enabled
            if should_show_window:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                env.render(mode='human', attn=attn)
                if args.record_video:
                    time.sleep(0.08)

        per_step_rewards[t] = episode_rewards/episode_steps
        num_success += info['n'][0]['is_success']
        episode_length = (episode_length*t + info['n'][0]['world_steps'])/(t+1)

        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.world.min_dists)
        elif args.env_name == 'simple_formation' or args.env_name=='simple_line':
            final_min_dists.append(env.world.dists)

        if should_show_window:
            print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['n'][0]['is_success'],
                per_step_rewards[t][0],info['n'][0]['world_steps']))
        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents

        # Save GIF for this episode
        if should_save_gif and frames:
            # Simple sequential naming within the evaluation folder
            gif_filename = f"{t+1}.gif"  # 1.gif, 2.gif, 3.gif, 4.gif, 5.gif
            gif_path = os.path.join(eval_folder, gif_filename)
            
            try:
                # Save frames as GIF using imageio
                imageio.mimsave(gif_path, frames, duration=0.1)  # 0.1s per frame (10 FPS)
                print(f"Saved GIF: {gif_path}")
            except Exception as e:
                print(f"Error saving GIF {gif_path}: {e}")
                
    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length = evaluate(args, args.seed, 
                    policies_list, ob_rms, args.render, render_attn=args.masking)
    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f})"
            .format(per_step_rewards.mean(0),num_success,args.num_eval_episodes,episode_length))
    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))
