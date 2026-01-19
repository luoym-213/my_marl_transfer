import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time
import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# Global counter to track evaluate function calls
_evaluate_call_count = 0

def add_text_to_frame(frame_array, step, num_retired, num_agents, num_visited, num_targets, agent_rewards=None):
    """
    在numpy数组图像上添加文字信息
    
    Args:
        frame_array: numpy array of shape (H, W, 3)
        step: 当前时间步
        num_retired: 已退役智能体数量
        num_agents: 总智能体数量
        num_visited: 已访问目标数量
        num_targets: 总目标数量
        agent_rewards: 每个智能体当前步的奖励，numpy array of shape (num_agents,)
    
    Returns:
        numpy array with text overlay
    """
    # 转换为PIL Image
    img = Image.fromarray(frame_array)
    draw = ImageDraw.Draw(img)
    
    # 设置字体（如果系统没有，PIL会使用默认字体）
    try:
        # macOS字体路径
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            # 备选字体
            font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 24)
            font_small = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 18)
            font_tiny = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 16)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
    
    # 计算需要的背景框高度（考虑奖励信息）
    base_height = 110
    reward_height = 25 * num_agents if agent_rewards is not None else 0
    total_height = base_height + reward_height
    
    # 绘制半透明背景框
    box_padding = 15
    box_x1 = img.width - 250
    box_y1 = 10
    box_x2 = img.width - 10
    box_y2 = 10 + total_height
    
    # 创建一个带透明度的图层
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(box_x1, box_y1), (box_x2, box_y2)], 
        fill=(40, 40, 40, 200)  # 半透明深灰色背景
    )
    
    # 将overlay合成到原图
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 绘制文字
    text_x = box_x1 + box_padding
    text_y = box_y1 + box_padding
    line_height = 28
    
    # Step信息
    draw.text((text_x, text_y), f"Step: {step}", fill=(255, 255, 255), font=font)
    
    # Retired信息
    draw.text((text_x, text_y + line_height), 
              f"Retired: {num_retired}/{num_agents}", 
              fill=(255, 200, 100), font=font_small)
    
    # Targets信息
    draw.text((text_x, text_y + line_height * 2), 
              f"Targets: {num_visited}/{num_targets}", 
              fill=(100, 255, 100), font=font_small)
    
    # 显示每个智能体的奖励
    if agent_rewards is not None:
        reward_y = text_y + line_height * 3
        draw.text((text_x, reward_y), 
                  "Rewards:", 
                  fill=(200, 200, 255), font=font_small)
        
        # 显示每个智能体的奖励
        for i, reward in enumerate(agent_rewards):
            agent_reward_y = reward_y + 20 + i * 25
            # 根据奖励正负设置颜色
            if reward > 0:
                color = (100, 255, 100)  # 绿色 - 正奖励
            elif reward < 0:
                color = (255, 100, 100)  # 红色 - 负奖励
            else:
                color = (200, 200, 200)  # 灰色 - 零奖励
            
            draw.text((text_x + 10, agent_reward_y), 
                      f"A{i}: {reward:+.2f}", 
                      fill=color, font=font_tiny)
    
    return np.array(img)

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
        # Set observation range for visualization
        if hasattr(args, 'mask_obs_dist'):
            env.world.mask_obs_dist = args.mask_obs_dist

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
    all_high_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_high_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    # TODO: provide support for recurrent policies and mask
    recurrent_hidden_states = None
    mask = None

    # world.dists at the end of episode for simple_spread
    final_min_dists = []
    num_success = 0
    episode_length = 0
    
    # 新增变量，用于计算成功回合的平均步数
    successful_steps_total = 0
    successful_episodes_count = 0
    
    # ⭐ 新增指标统计
    all_time_to_cover_1 = []  # 覆盖第1个landmark的时间
    all_time_to_cover_2 = []  # 覆盖第2个landmark的时间
    all_time_to_cover_3 = []  # 覆盖第3个landmark的时间
    all_time_to_discover_all = []  # 发现所有landmark的时间

    # Create evaluation-specific folder if record_video is enabled
    eval_folder = None
    if args.record_video:
        eval_folder = os.path.join(args.gif_save_path, f"evaluation_{current_eval_call}")
        os.makedirs(eval_folder, exist_ok=True)
        print(f"GIF files will be saved to: {eval_folder}")

    for t in range(num_eval_episodes):
        obs, env_states, info = env.reset()
        master.envs_info = info
        # 修复：使用args.device确保设备一致性
        recurrent_hidden_states = torch.zeros(args.num_agents, args.recurrent_hidden_state_size, device=args.device)
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        masks = torch.ones(env.n, 1, device=args.device)
        episode_rewards = np.full(env.n, 0.0)
        episode_high_rewards = np.full(env.n, 0.0)
        episode_steps = 0
        
        # Determine rendering behavior
        # If record_video is True, we always want to save GIF for first 5 episodes
        # If render is True, we show the window for first 5 episodes
        should_save_gif = args.record_video and t < 5
        should_show_window = render and t < 5
        
        # Initialize frame collection for GIF
        frames = []
        # Track statistics for each frame
        frame_stats = []  # List of tuples: (step, num_retired, num_visited)
        
        # Initialize goals to None at the start of each episode，初始化为tensor,0
        goals = torch.zeros((len(obs), 2), dtype=torch.float32, device=args.device)
        tasks = torch.zeros((len(obs), 1), dtype=torch.long, device=args.device)
        landmark_data = torch.zeros((len(obs), args.num_agents, 4), dtype=torch.float32, device=args.device)
        landmark_mask = torch.zeros((len(obs), args.num_agents, 1), dtype=torch.float32, device=args.device)
        
        # ⭐ 跟踪发现和覆盖状态
        discovered_landmarks = set()  # 已发现的landmark索引
        covered_landmarks = set()  # 已覆盖的landmark索引
        time_to_discover_all = None  # 发现所有landmark的时间
        time_to_cover_1 = None  # 覆盖第1个landmark的时间
        time_to_cover_2 = None  # 覆盖第2个landmark的时间
        time_to_cover_3 = None  # 覆盖第3个landmark的时间

        # Initial render for GIF saving (if needed)
        if should_save_gif:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            render_result = env.render(
                mode='rgb_array', 
                attn=attn,
                goals=goals,
                show_voronoi=True,
                show_uncertainty=True  # 👈 启用不确定性显示
            )
            if render_result:
                frames.append(render_result[0])
                # 记录初始统计信息
                num_retired = sum(env.agents_done) if hasattr(env, 'agents_done') else 0
                num_visited = sum(env.landmark_visited) if hasattr(env, 'landmark_visited') else 0
                num_targets = len(env.world.landmarks) if hasattr(env.world, 'landmarks') else args.num_agents
                # 初始帧没有奖励，使用零数组
                initial_rewards = np.zeros(args.num_agents)
                frame_stats.append((0, num_retired, num_visited, num_targets, initial_rewards))
        
        # Show window if render is enabled
        if should_show_window:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            env.render(
                mode='human', 
                attn=attn,
                goals=goals,
                show_voronoi=True,
                show_uncertainty=True  # 👈 启用不确定性显示
            )
        
        while not np.all(done):
            actions = []
            with torch.no_grad():
                # print("step:", info['world_steps'])
                actions, goals, tasks, landmark_data, landmark_mask, graph_data = master.eval_act(obs, env_states, masks,
                                                                                      goals, tasks, 
                                                                                      landmark_data, 
                                                                                      landmark_mask)
            episode_steps += 1
            step_data = {'agents_actions': actions, 'agents_goals': goals, 'agents_tasks': tasks, 'graph_data': graph_data} 
            if isinstance(step_data['agents_goals'], torch.Tensor):
                step_data['agents_goals'] = step_data['agents_goals'].cpu().numpy()
            if isinstance(step_data['agents_tasks'], torch.Tensor):
                step_data['agents_tasks'] = step_data['agents_tasks'].cpu().numpy()
            obs, reward, high_reward, done_info, info, env_states = env.step(step_data)
            done = done_info['agent']
            done_agent = np.array(done_info['agent'])
            high_reward = torch.from_numpy(np.stack(high_reward)).float().to(args.device)
            masks = torch.FloatTensor(1-1.0*done_agent).to(args.device)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device)
            obs = normalize_obs(obs, obs_mean, obs_std)
            master.envs_info = info
            episode_rewards += reward.cpu().numpy()
            episode_high_rewards += high_reward.cpu().numpy()
            
            # ⭐ 跟踪发现状态（基于sensor_range内的检测）
            if hasattr(args, 'mask_obs_dist'):
                sensor_range = args.mask_obs_dist
                for lm_idx, landmark in enumerate(env.world.landmarks):
                    lm_pos = landmark.state.p_pos
                    for agent in env.agents:
                        if np.linalg.norm(agent.state.p_pos - lm_pos) < sensor_range:
                            discovered_landmarks.add(lm_idx)
                            break
            
            # ⭐ 记录发现所有landmark的时间（首次）
            if time_to_discover_all is None and len(discovered_landmarks) == len(env.world.landmarks):
                time_to_discover_all = episode_steps
            
            # ⭐ 跟踪覆盖状态
            if hasattr(env, 'visited_landmarks'):
                current_covered = set(env.visited_landmarks) if isinstance(env.visited_landmarks, list) else env.visited_landmarks
            elif hasattr(env, 'landmark_visited'):
                current_covered = set([i for i, v in enumerate(env.landmark_visited) if v])
            else:
                current_covered = set()
            
            # ⭐ 更新覆盖状态并记录时间戳
            new_covered = current_covered - covered_landmarks
            if new_covered:
                covered_landmarks.update(new_covered)
                num_covered = len(covered_landmarks)
                
                # 记录覆盖第1/2/3个landmark的时间
                if num_covered == 1 and time_to_cover_1 is None:
                    time_to_cover_1 = episode_steps
                elif num_covered == 2 and time_to_cover_2 is None:
                    time_to_cover_2 = episode_steps
                elif num_covered == 3 and time_to_cover_3 is None:
                    time_to_cover_3 = episode_steps
            
            # Render for GIF saving (if needed)
            if should_save_gif:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                render_result = env.render(
                    mode='rgb_array', 
                    attn=attn,
                    goals=step_data['agents_goals'],
                    show_voronoi=True,
                    show_uncertainty=True,  # 👈 启用不确定性显示
                    tasks=step_data['agents_tasks'],
                    info=info,
                    graph_data=step_data.get('graph_data')  # 传递图数据
                )
                if render_result:
                    frames.append(render_result[0])
                    # 记录当前步的统计信息和奖励
                    num_retired = sum(env.agents_done) if hasattr(env, 'agents_done') else 0
                    num_visited = sum(env.landmark_visited) if hasattr(env, 'landmark_visited') else 0
                    num_targets = len(env.world.landmarks) if hasattr(env.world, 'landmarks') else args.num_agents
                    # 记录当前步每个智能体的奖励（high-level）
                    current_step_rewards = high_reward.cpu().numpy()
                    frame_stats.append((episode_steps, num_retired, num_visited, num_targets, current_step_rewards))
            
            # Show window if render is enabled
            if should_show_window:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                env.render(
                    mode='human', 
                    attn=attn,
                    goals=step_data['agents_goals'],
                    show_voronoi=True,
                    show_uncertainty=True,  # 👈 启用不确定性显示
                    tasks=step_data['agents_tasks'],
                    info=info
                )
                if args.record_video:
                    time.sleep(0.08)

        per_step_rewards[t] = episode_rewards/episode_steps
        per_high_step_rewards[t] = episode_high_rewards/episode_steps
        num_success += info['is_success']
        episode_length = (episode_length*t + info['n'][0]['world_steps'])/(t+1)
        
        # 更新成功回合的统计数据
        if info['is_success']:
            successful_steps_total += info['n'][0]['world_steps']
            successful_episodes_count += 1
        
        # ⭐ 记录新增指标
        all_time_to_cover_1.append(time_to_cover_1 if time_to_cover_1 is not None else episode_steps)
        all_time_to_cover_2.append(time_to_cover_2 if time_to_cover_2 is not None else episode_steps)
        all_time_to_cover_3.append(time_to_cover_3 if time_to_cover_3 is not None else episode_steps)
        all_time_to_discover_all.append(time_to_discover_all if time_to_discover_all is not None else episode_steps)

        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.world.min_dists)
        elif args.env_name == 'simple_formation' or args.env_name=='simple_line':
            final_min_dists.append(env.world.dists)

        if should_show_window:
            print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['is_success'],
                per_step_rewards[t][0],info['n'][0]['world_steps']))
        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents
        all_high_episode_rewards[t, :] = episode_high_rewards # all_episode_rewards shape: num_eval_episodes x num agents

        # Save GIF for this episode
        if should_save_gif and frames:
            # Simple sequential naming within the evaluation folder
            gif_filename = f"{t+1}.gif"  # 1.gif, 2.gif, 3.gif, 4.gif, 5.gif
            gif_path = os.path.join(eval_folder, gif_filename)
            
            try:
                # Add text overlay to all frames
                frames_with_text = []
                for i, (frame, stats) in enumerate(zip(frames, frame_stats)):
                    step, num_retired, num_visited, num_targets, agent_rewards = stats
                    frame_with_text = add_text_to_frame(
                        frame, 
                        step, 
                        num_retired, 
                        args.num_agents, 
                        num_visited, 
                        num_targets,
                        agent_rewards
                    )
                    frames_with_text.append(frame_with_text)
                
                # Save frames as GIF using imageio
                imageio.mimsave(gif_path, frames_with_text, duration=0.1)  # 0.1s per frame (10 FPS)
                print(f"Saved GIF: {gif_path}")
            except Exception as e:
                print(f"Error saving GIF {gif_path}: {e}")
    
    # 计算成功回合的平均步数
    successful_average_length = 0
    if successful_episodes_count > 0:
        successful_average_length = successful_steps_total / successful_episodes_count

    return (all_episode_rewards, per_step_rewards, all_high_episode_rewards, per_high_step_rewards, 
            final_min_dists, num_success, episode_length, successful_average_length, successful_episodes_count,
            all_time_to_cover_1, all_time_to_cover_2, all_time_to_cover_3, all_time_to_discover_all)


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    (all_episode_rewards, per_step_rewards, all_high_episode_rewards, per_high_step_rewards, 
     final_min_dists, num_success, episode_length, successful_average_length, successful_episodes_count,
     all_time_to_cover_1, all_time_to_cover_2, all_time_to_cover_3, all_time_to_discover_all) = evaluate(
        args, args.seed, policies_list, ob_rms, args.render, render_attn=args.masking)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Per Step Reward: {per_step_rewards.mean(0)}")
    print(f"\n【任务完成率】")
    print(f"  Success Rate: {num_success/args.num_eval_episodes*100:.1f}% ({num_success}/{args.num_eval_episodes})")
    print(f"  Successful Episodes Avg Length: {successful_average_length:.2f} ({successful_episodes_count}/{args.num_eval_episodes})")
    print(f"\n【覆盖时间】")
    print(f"  Cover 1st Landmark: {np.mean(all_time_to_cover_1):.2f} ± {np.std(all_time_to_cover_1):.2f} steps")
    print(f"  Cover 2nd Landmark: {np.mean(all_time_to_cover_2):.2f} ± {np.std(all_time_to_cover_2):.2f} steps")
    print(f"  Cover 3rd Landmark: {np.mean(all_time_to_cover_3):.2f} ± {np.std(all_time_to_cover_3):.2f} steps")
    print(f"\n【发现时间】")
    print(f"  Discover All Landmarks: {np.mean(all_time_to_discover_all):.2f} ± {np.std(all_time_to_discover_all):.2f} steps")
    print(f"\n【总体性能】")
    print(f"  Average Episode Length: {episode_length:.2f} steps")
    if final_min_dists:
        print(f"  Final Min Dists: {np.stack(final_min_dists).mean(0)}")
    print("="*60)
