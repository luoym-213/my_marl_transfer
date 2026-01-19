"""
Evaluate ACO-MTS algorithm in MPE environment
Adapted from eval_baseline.py to test ACO planner
"""
import numpy as np
import torch
import os
import imageio
from utils import normalize_obs
from PIL import Image, ImageDraw, ImageFont
from arguments import get_args
from learner import setup_master
from src.baseline.aco_mts_planner_mpe import Config, Environment, ACOPlanner, BeliefMap


def add_text_to_frame(frame_array, step, num_visited, num_targets, et_value=None):
    """Add text overlay to frame"""
    img = Image.fromarray(frame_array)
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Background box
    box_padding = 15
    box_x1 = img.width - 280
    box_y1 = 10
    box_x2 = img.width - 10
    box_y2 = 130
    
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(box_x1, box_y1), (box_x2, box_y2)], 
        fill=(40, 40, 40, 200)
    )
    
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw text
    text_x = box_x1 + box_padding
    text_y = box_y1 + box_padding
    line_height = 28
    
    draw.text((text_x, text_y), f"Step: {step}", fill=(255, 255, 255), font=font)
    draw.text((text_x, text_y + line_height), 
              f"Targets: {num_visited}/{num_targets}", 
              fill=(100, 255, 100), font=font_small)
    
    if et_value is not None:
        draw.text((text_x, text_y + line_height * 2), 
                  f"ET: {et_value:.2f}", 
                  fill=(200, 200, 255), font=font_small)
    
    return np.array(img)


def evaluate_aco_mts(args, seed=None, render=True, num_eval_episodes=5, policies_list=None, ob_rms=None):
    """
    Evaluate ACO-MTS algorithm in MPE environment
    
    Args:
        args: Configuration arguments
        seed: Random seed
        render: Whether to save GIF animations
        num_eval_episodes: Number of episodes to evaluate
        policies_list: Pretrained policy parameters (optional, for landmark collection)
        ob_rms: Observation running mean/std (optional)
    """
    print("\n" + "="*60)
    print("ACO-MTS Evaluation in MPE Environment")
    print("="*60)
    
    # Set seed
    if seed is None:
        seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Evaluation Seed: {seed}")
    
    # Setup environment
    master, env = setup_master(args, return_env=True)
    env.seed(seed)
    
    # ⭐ 加载底层策略参数到 master（如果提供）
    if policies_list is not None:
        print("Loading pretrained policies into agents...")
        master.load_models(policies_list)
        master.set_eval_mode()
        print("✓ Policies loaded and set to evaluation mode\n")
    else:
        print("⚠ Running with untrained/random low-level policy\n")
    
    # Force continuous action space for ACO planner output
    env.discrete_action_input = False
    env.discrete_action_space = False
    env.force_discrete_action = False
    # Note: agent.accel defaults to 5.0 (sensitivity in environment.py)
    # Actions will be multiplied by this sensitivity factor
    
    # Configure ACO-MTS parameters based on environment
    # MPE Physics: Force=5.0 -> Max Speed~2.0 -> Max Dist/Step~0.2 (dt=0.1)
    # We set max_speed to 0.15 to be conservative (accounting for inertia/acceleration)
    config = Config(
        world_size=2.0,
        grid_size=100,
        num_targets=args.num_agents,
        num_uavs=args.num_agents,
        max_speed=0.15,  # Adjusted to match MPE physics limits
        dt=0.1,
        num_ants=5,
        num_iterations=1,
        sensor_range=args.mask_obs_dist if hasattr(args, 'mask_obs_dist') else 0.3,
        sensor_fidelity=0.8,
        num_action_directions=8
    )
    
    print(f"Configuration:") 
    print(f"  Num Agents/Targets: {config.num_uavs}/{config.num_targets}")
    print(f"  Sensor Range: {config.sensor_range}")
    print(f"  Grid Size: {config.grid_size}x{config.grid_size}")
    print(f"  ACO Iterations: {config.num_iterations}")
    print(f"  Ants per Iteration: {config.num_ants}")
    
    # Create output directory
    eval_folder = os.path.join(args.gif_save_path, "aco_mts_evaluation")
    os.makedirs(eval_folder, exist_ok=True)
    print(f"  Output folder: {eval_folder}\n")
    
    # Statistics
    all_et_values = []
    all_success_rates = []
    all_steps = []
    
    for episode in range(num_eval_episodes):
        print(f"\n--- Episode {episode + 1}/{num_eval_episodes} ---")
        
        # Reset environment
        obs, env_states, info = env.reset()
        master.envs_info = info
        
        # Get target positions from environment
        if hasattr(env.world, 'landmarks'):
            targets = [landmark.state.p_pos.copy() for landmark in env.world.landmarks]
        else:
            # Fallback: random positions
            targets = [np.random.uniform(-0.8, 0.8, 2) for _ in range(config.num_targets)]
        
        print(f"Target positions: {[f'({t[0]:.2f}, {t[1]:.2f})' for t in targets]}")
        
        # Create ACO environment and planner
        aco_env = Environment(config, targets)
        
        # Get UAV starting positions from environment
        if hasattr(env, 'agents'):
            uav_positions = [agent.state.p_pos.copy() for agent in env.agents]
        else:
            # Fallback: corners
            uav_positions = [
                np.array([-0.9, -0.9]),
                np.array([0.9, 0.9])
            ][:config.num_uavs]
        
        planner = ACOPlanner(config, aco_env)
        planner.uav_positions = uav_positions
        
        print(f"UAV start positions: {[f'({u[0]:.2f}, {u[1]:.2f})' for u in uav_positions]}")
        
        # Run ACO optimization
        print("Running ACO optimization...")
        et_history = planner.optimize()
        final_et = et_history[-1]
        all_et_values.append(final_et)
        
        print(f"Optimization complete: ET = {final_et:.4f}")
        
        # Execute best solution in environment
        if planner.best_solution is None:
            print("Warning: No solution found!")
            continue
        
        # ⭐ 设置观测归一化参数
        if ob_rms is not None:
            obs_mean = ob_rms[0].mean
            obs_std = np.sqrt(ob_rms[0].var + 1e-8)
        else:
            obs_mean = None
            frames = []
        step = 0
        num_visited = 0
        done = [False] * env.n
        masks = torch.ones(env.n, 1, device=args.device)
        
        # Get maximum trajectory length
        max_traj_len = max(len(traj) for traj in planner.best_solution.values())
        
        print(f"Executing solution (max {max_traj_len} steps)...")
        obs_mean = None
        obs_std = None

        # Initialize goals to None at the start of each episode，初始化为tensor,0
        goals = torch.zeros((len(obs), 2), dtype=torch.float32, device=args.device)
        tasks = torch.zeros((len(obs), 1), dtype=torch.long, device=args.device)
        landmark_data = torch.zeros((len(obs), args.num_agents, 4), dtype=torch.float32, device=args.device)
        landmark_mask = torch.zeros((len(obs), args.num_agents, 1), dtype=torch.float32, device=args.device)
        
        # Execute trajectory
        for t in range(max_traj_len):
            # 在这里收集是否观测到landmark的数据，如果有观测到则更新belief map
            # 按照最贪婪策略，如果观测到landmark则认为已经发现，立刻执行覆盖策略，安排最近的agent前往覆盖
            # 否则持续执行原有路径规划
            # 1. 获取 RL 策略的输出 (重命名为 rl_actions 以避免冲突)
            rl_actions, goals, tasks, landmark_data, landmark_mask = master.eval_base_act(obs, env_states, masks,
                                                                                    goals, tasks, 
                                                                                    landmark_data, 
                                                                                    landmark_mask)
            
            # 2. 初始化环境动作数组 (连续动作空间: [num_agents, 2])
            actions = np.zeros((len(env.agents), 2), dtype=np.float32)


            # Get planned positions for all UAVs at time t
            planned_positions = []
            for uav_id in range(config.num_uavs):
                if t < len(planner.best_solution[uav_id]):
                    planned_positions.append(planner.best_solution[uav_id][t])
                else:
                    planned_positions.append(planner.best_solution[uav_id][-1])
            
            # Convert to actions (velocity to reach planned position)
            # Match environment's velocity scale: base_velocity * sensitivity(5.0)
            # We provide base velocity (unit direction scaled), environment multiplies by 5.0
            for i, agent in enumerate(env.agents):
                if i < len(planned_positions) and not tasks[i]:
                    target_pos = planned_positions[i]
                    current_pos = agent.state.p_pos
                    direction = target_pos - current_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance > 1e-5:
                        # Compute unit direction
                        direction = direction / distance
                        
                        # Apply max action (1.0) to track the plan
                        # Scale down if very close to avoid overshoot
                        # MPE applies Force = Action * 5.0
                        scale = min(1.0, distance / config.max_speed)
                        actions[i] = direction * scale
                    else:
                        actions[i] = np.zeros(2)
                elif tasks[i]:
                    # 如果任务是 collect (tasks[i]=1)，使用 RL 策略的动作
                    # 由于环境被强制为连续输入，我们需要将 RL 的离散动作索引转换为速度向量
                    # MPE 默认离散映射: 1=Left, 2=Right, 3=Down, 4=Up
                    idx = int(rl_actions[i])
                    u = np.zeros(2)
                    if idx == 1: u[0] = -1.0
                    elif idx == 2: u[0] = +1.0
                    elif idx == 3: u[1] = -1.0
                    elif idx == 4: u[1] = +1.0
                    actions[i] = u
            
            # Step environment
            # Ensure goals and tasks are numpy arrays for the environment interaction
            goals_np = goals.cpu().numpy() if isinstance(goals, torch.Tensor) else goals
            tasks_np = tasks.cpu().numpy() if isinstance(tasks, torch.Tensor) else tasks

            print(f"Step {step + 1}: Actions: {actions},\n Goals: {goals_np},\n Tasks: {tasks_np.flatten()}")
            
            step_data = {'agents_actions': actions, 'agents_goals': goals_np, 'agents_tasks': tasks_np} 
            
            obs, reward, high_reward, done_info, info, env_states = env.step(step_data)
            done = done_info['agent']
            done_agent = np.array(done_info['agent'])
            masks = torch.FloatTensor(1-1.0*done_agent).to(args.device)
            obs = normalize_obs(obs, obs_mean, obs_std)
            master.envs_info = info
            step += 1
            
            # Count visited targets
            if hasattr(env, 'visited_landmarks'):
                num_visited = len(env.visited_landmarks)
            elif hasattr(env, 'landmark_visited'):
                num_visited = sum(env.landmark_visited)
            
            # Render frame
            if render and episode < 5:  # Save GIF for first 5 episodes
                render_result = env.render(
                    mode='rgb_array',
                    goals=np.array(planned_positions),
                    show_voronoi=True,
                    show_uncertainty=True
                )
                if render_result:
                    frame_with_text = add_text_to_frame(
                        render_result[0],
                        step,
                        num_visited,
                        config.num_targets,
                        final_et
                    )
                    frames.append(frame_with_text)
            
            # Check termination
            if np.all(done) or info.get('is_success', False):
                break
        
        # Record statistics
        all_steps.append(step)
        is_success = info.get('is_success', False)
        all_success_rates.append(1.0 if is_success else 0.0)
        
        print(f"Episode complete: Steps={step}, Success={is_success}, Visited={num_visited}/{config.num_targets}")
        
        # Save GIF
        if render and episode < 5 and frames:
            gif_path = os.path.join(eval_folder, f"episode_{episode+1}.gif")
            imageio.mimsave(gif_path, frames, duration=0.1)
            print(f"Saved GIF: {gif_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes: {num_eval_episodes}")
    print(f"Average ET: {np.mean(all_et_values):.4f} ± {np.std(all_et_values):.4f}")
    print(f"Success Rate: {np.mean(all_success_rates)*100:.1f}%")
    print(f"Average Steps: {np.mean(all_steps):.2f} ± {np.std(all_steps):.2f}")
    print("="*60)
    
    return {
        'et_values': all_et_values,
        'success_rates': all_success_rates,
        'steps': all_steps
    }


if __name__ == '__main__':
    args = get_args()
    
    # Override some args for ACO evaluation
    if not hasattr(args, 'num_eval_episodes'):
        args.num_eval_episodes = 5
    if not hasattr(args, 'record_video'):
        args.record_video = True
    if not hasattr(args, 'gif_save_path'):
        args.gif_save_path = 'results'
    
    # ⭐ 加载底层策略网络参数（可选）
    policies_list = None
    ob_rms = None
    if hasattr(args, 'load_dir') and args.load_dir is not None:
        print(f"\n{'='*60}")
        print(f"Loading pretrained low-level policy from: {args.load_dir}")
        print(f"{'='*60}\n")
        try:
            checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
            policies_list = checkpoint['models']
            ob_rms = checkpoint.get('ob_rms', None)  # 可能没有 ob_rms
            print("✓ Successfully loaded checkpoint")
            print(f"  - Models: {len(policies_list)} policies")
            print(f"  - Observation normalization: {'Yes' if ob_rms else 'No'}\n")
        except Exception as e:
            print(f"⚠ Warning: Failed to load checkpoint: {e}")
            print("  Continuing without pretrained policy...\n")
            policies_list = None
            ob_rms = None
    else:
        print("\n⚠ No pretrained policy specified (use --load-dir to load)")
        print("  Using random low-level policy for landmark collection\n")
    
    results = evaluate_aco_mts(
        args, 
        seed=args.seed if hasattr(args, 'seed') else None,
        render=args.record_video,
        num_eval_episodes=args.num_eval_episodes,
        policies_list=policies_list,
        ob_rms=ob_rms
    )
