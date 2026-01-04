import numpy as np
import torch
from rlcore.algo import IPPO, JointPPO
from rlagent import Neo
from mpnn import MPNN
from utils import make_multiagent_env


def setup_master(args, env=None, return_env=False):
    if env is None:
        env = make_multiagent_env(args.env_name, num_agents=args.num_agents, dist_threshold=args.dist_threshold, 
                                  arena_size=args.arena_size, identity_size=args.identity_size,
                                  mask_obs_dist=args.mask_obs_dist if hasattr(args, 'mask_obs_dist') else None)
    policy1 = None
    policy2 = None
    team1 = []
    team2 = []

    num_adversary = 0
    num_friendly = 0
    for i,agent in enumerate(env.world.policy_agents):
        if hasattr(agent, 'adversary') and agent.adversary:
            num_adversary += 1
        else:
            num_friendly += 1

    # share a common policy in a team
    action_space = env.action_space[i]
    entity_mp = args.entity_mp
    if args.env_name == 'simple_spread':
        num_entities = args.num_agents
    elif args.env_name == 'simple_formation':
        num_entities = 1
    elif args.env_name == 'simple_line':
        num_entities = 2
    else:
        raise NotImplementedError('Unknown environment, define entity_mp for this!')

    if entity_mp:
        pol_obs_dim = env.observation_space[i].shape[0] - 2*(2*num_entities-1)
    else:
        pol_obs_dim = env.observation_space[i].shape[0]
    
    '''
    print("action_space: ", action_space)
    print("observation_space:", env.observation_space[i].shape[0])
    print("pol_obs_dim: ", pol_obs_dim)
    '''

    # index at which agent's position is present in its observation
    pos_index = args.identity_size + 2
    for i, agent in enumerate(env.world.policy_agents):
        obs_dim = env.observation_space[i].shape[0]

        if hasattr(agent, 'adversary') and agent.adversary:
            if policy1 is None:
                policy1 = MPNN(input_size=pol_obs_dim,num_agents=num_adversary,num_entities=num_entities,action_space=action_space,
                               pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp, is_recurrent=args.is_recurrent).to(args.device)
            team1.append(Neo(args,policy1,(obs_dim,),action_space))
        else:
            if policy2 is None:
                policy2 = MPNN(input_size=pol_obs_dim,num_agents=num_friendly,num_entities=num_entities,action_space=action_space,
                               pos_index=pos_index, mask_dist=args.mask_dist,mask_obs_dist=args.mask_obs_dist,entity_mp=entity_mp,is_recurrent=args.is_recurrent).to(args.device)

            team2.append(Neo(args,policy2,(obs_dim,),action_space))

        # ================== 新增：加载预训练底层网络 ==================
        # 假设你在 args 中定义了 load_low_level_path 和 freeze_low_level
        if hasattr(args, 'load_low_level_path') and args.load_low_level_path is not None:
            print(f"🔄 Loading pretrained low-level model from: {args.load_low_level_path}")
            
            # 如果有 policy1 (Adversary)，加载
            if policy1 is not None:
                policy1.load_pretrained_low_level(args.load_low_level_path, freeze=True) # 建议默认冻结
                
            # 如果有 policy2 (Friendly)，加载
            if policy2 is not None:
                policy2.load_pretrained_low_level(args.load_low_level_path, freeze=True)
        # ============================================================
        
    master = Learner(args, [team1, team2], [policy1, policy2], env=env) # 传入并行环境
    
    if args.continue_training:
        print("Loading pretrained model")
        master.load_models(torch.load(args.load_dir, map_location=torch.device('cpu'))['models'])

    if return_env:
        return master, env
    return master


class Learner(object):
    def __init__(self, args, teams_list, policies_list, env):
        self.teams_list = [x for x in teams_list if len(x)!=0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]
        # self.trainers_list = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        #                                args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
        #                                use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        # ⭐ 检测是否加载了预训练底层网络
        self.use_pretrained_low_level = (
            hasattr(args, 'load_low_level_path') and 
            args.load_low_level_path is not None
        )

        # 初始化训练器
        if not self.use_pretrained_low_level:
            self.trainers_list = [IPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                                        args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                        use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        self.high_trainers_list = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        self.device = args.device
        self.env = env
        self.envs_info = None
        self.high_level_interval = args.high_level_interval

        self.top_k = args.top_k
        self.rrt_max_iter = args.rrt_max_iter

    @property
    def all_policies(self):
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        return self.policies_list[0].attn_mat

    def initialize_obs(self, obs):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[:,i,:]).float().to(self.device))
            agent.rollouts.to(self.device)

    def initialize_env_state(self, env_state):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_env_state(torch.from_numpy(env_state).float().to(self.device))
            agent.rollouts.to(self.device)

    def act(self, step):
        # 根据当前的环境中智能体的状态进行决策，生成下一步的动作列表，这里可以直接当作分层网络的总启，下面再进行细分是high还是low
        actions_list = []
        goals_list = []
        tasks_list = []
        for team, policy in zip(self.teams_list, self.policies_list):
            # concatenate all inputs
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])
            all_env_state = torch.cat([agent.rollouts.env_states[step] for agent in team]) # 实际上只需要其中一个agent的环境状态就够了，即all_env_state[:num_processes]

            # 默认采取之前的目标分配，[num_agents * num_processes, 2]，以及任务类型[num_agents * num_processes, 1]
            # 即使step=0时，也复制goals[-1]的值，保证goal有效
            all_goals = torch.cat([agent.rollouts.goals[step-1] for agent in team])
            all_tasks = torch.cat([agent.rollouts.tasks[step-1] for agent in team])
            all_higoal_log_probs = torch.cat([agent.rollouts.higoal_log_probs[step-1] for agent in team])
            all_landmark_datas = torch.cat([agent.rollouts.landmark_datas[step-1] for agent in team])
            all_landmark_masks = torch.cat([agent.rollouts.landmark_masks[step-1] for agent in team])

            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # ⭐ 构建Critic输入
            entropy_maps = torch.stack([torch.from_numpy(np.array(info['entropy_map'])).float() 
                                        for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            heatmaps = torch.stack([torch.from_numpy(np.array(info['heatmap'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            landmark_heatmaps = torch.stack([torch.from_numpy(np.array(info['landmark_heatmap'])).float() 
                                            for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            all_critic_map_inp = torch.stack([entropy_maps, heatmaps, landmark_heatmaps], dim=1)  # [num_processes, 3, H, W]
            
            # ⭐ 收集goal_done和battery信息
            goal_done_list = [info['goal_done'] for info in self.envs_info]
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)
            agent_world_steps = torch.tensor(
                [info['world_steps'] for info in self.envs_info], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(1).repeat(1, num_agents).unsqueeze(-1)
            agent_batterys = (50.0 - agent_world_steps) / 50.0

            # ⭐ 生成landmark节点
            detected_maps = [torch.from_numpy(np.array(info['map'][1])).float().to(self.device) 
                            for info in self.envs_info]
            new_detected, new_detected_masks = self.update_landmark_info(all_landmark_datas, all_landmark_masks, detected_maps, self.device)

            # ⭐ 准备智能体节点数据
            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)
            voronoi_masks = torch.stack([
                torch.stack([torch.from_numpy(np.array(info['voronoi_masks'][a])).float() 
                            for a in range(num_agents)]) 
                for info in self.envs_info
            ]).to(self.device)
            agent_positions = all_obs[:, 2:4].view(num_agents, num_processes, 2).transpose(0, 1)
            agent_vels = all_obs[:, 0:2].view(num_agents, num_processes, 2).transpose(0, 1)
            agent_goals = all_goals.view(num_agents, num_processes, 2).transpose(0, 1)
            agent_nodes = torch.cat([agent_positions, agent_goals], dim=-1)
            ego_nodes = torch.cat([agent_positions, agent_vels, agent_batterys], dim=-1)

            # ⭐ 高层决策（如果需要）
            if goal_done_mask.any():
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_entropy_maps[proc_indices],
                    voronoi_masks[proc_indices]
                ], dim=0)
                vec_inp_agents = agent_nodes[proc_indices]
                
                # ⭐ RRT生成探索节点
                batch_explore_nodes = policy.get_explore_nodes(self.top_k, self.rrt_max_iter, vec_inp_agents, map_inps, agent_indices)
                batch_explore_nodes = batch_explore_nodes.reshape(-1, batch_explore_nodes.shape[-2], batch_explore_nodes.shape[-1])
                
                # ⭐ 获取landmark节点
                batch_ego_nodes = ego_nodes[proc_indices, agent_indices]
                linear_indices = agent_indices * num_processes + proc_indices
                batch_landmark_nodes, batch_landmark_node_masks = policy.get_landmark_nodes(
                    all_obs[:,2:4], new_detected, new_detected_masks, linear_indices
                )
                
                # ⭐ 获取边特征
                batch_ego_to_explore_edges, batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks = policy.get_edge_features(
                    batch_explore_nodes, batch_landmark_nodes, batch_landmark_node_masks
                )

                # ⭐ 高层策略决策
                batch_goals = policy.get_high_level_goal(
                    batch_ego_nodes, # Tensor shape [N, 5]
                    batch_explore_nodes, # Tensor shape [N, K, 4]
                    batch_ego_to_explore_edges, # Tensor [N, K, 3], length N
                    batch_landmark_nodes, # tensor: [N, Max_L, 4]
                    batch_landmark_node_masks, # tensor: [N, Max_L, 1]
                    batch_ego_to_landmark_edges, # Tensor [N, Max_L, 3]
                    batch_ego_to_landmark_edge_masks # Tensor [N, Max_L, 1]
                )
                
                # ⭐ 更新目标和landmark
                all_goals[linear_indices] = batch_goals["waypoints"]
                all_tasks[linear_indices] = batch_goals["action_modes"]
                all_higoal_log_probs[linear_indices] = batch_goals["node_log_probs"]
                
                for i, lin_idx in enumerate(linear_indices):
                    if batch_goals["action_modes"][i, 0] == 1:  # 选择的是 landmark
                        # 获取选中的 waypoint（绝对世界坐标）
                        selected_waypoint = batch_goals["waypoints"][i]  # [2]
                        
                        # 在对应的 landmark 列表中找到匹配的 landmark
                        landmarks = new_detected[lin_idx]  # [max_landmarks, 4]
                        landmark_mask = new_detected_masks[lin_idx]  # [max_landmarks, 1]
                        
                        # 找到有效的 landmark
                        valid_mask = landmark_mask[:, 0] > 0.5
                        if valid_mask.any():
                            # 计算距离（使用绝对坐标）
                            landmark_positions = landmarks[:, :2]  # [max_landmarks, 2]
                            distances = torch.norm(landmark_positions - selected_waypoint, dim=1)  # [max_landmarks]
                            distances = distances.masked_fill(~valid_mask, float('inf'))
                            
                            # 找到最近的 landmark
                            min_idx = distances.argmin()
                            if distances[min_idx] < 0.05:  # 匹配阈值 0.05
                                new_detected[:, min_idx, 3] = 1.0  # 设置 is_targeted = 1
                
                # 更新所有智能体的 landmark 数据
                all_landmark_datas = new_detected
                all_landmark_masks = new_detected_masks

            # ⭐ 准备rollout数据
            all_ego_nodes = ego_nodes.view(num_processes * num_agents, -1)
            K = self.top_k
            all_explore_nodes = torch.zeros(num_processes * num_agents, K, 4, device=self.device)
            if goal_done_mask.any():
                # 将决策智能体的 explore nodes 填充到对应位置
                all_explore_nodes[linear_indices] = batch_explore_nodes  # batch_explore_nodes: [N, K, 4]
            
            # 8. 计算高层value
            # Vector Input: Tensor shape [Batch, N_agents, 4] [x,y,x_g,y_g]
            # Map Input: Tensor shape [Batch, 3, H, W]。
            # all_critic_vec_inp = [x,y,x_g,y_g]
            ## 拼接成[num_processes, num_agents, 4]
            all_critic_nodes = torch.cat([agent_positions, all_goals.view(num_agents, num_processes, 2).transpose(0, 1)], dim=-1)  # [num_processes, num_agents, 4]
            all_high_value = policy.get_high_value(all_critic_map_inp, all_critic_nodes) # 计算所有process的高层value： [num_processes, num_agents]

            # ⭐ 底层策略
            props = policy.low_level_act(all_obs, all_goals, deterministic=False)

            # ⭐ 拆分和存储结果
            n = len(team)
            
            # 拆分图数据
            all_ego_nodes_split = torch.chunk(all_ego_nodes, n)
            all_explore_nodes_split = torch.chunk(all_explore_nodes, n)
            
            # 拆分其他数据
            all_goals = torch.chunk(all_goals, n)
            all_tasks = torch.chunk(all_tasks, n)
            all_higoal_log_probs = torch.chunk(all_higoal_log_probs, n)
            all_landmark_datas_split = torch.chunk(all_landmark_datas, n)
            all_landmark_masks_split_tensor = torch.chunk(all_landmark_masks, n)
            all_high_value = torch.chunk(all_high_value, n, dim=1)
            all_value, all_action, all_action_log_prob = [torch.chunk(x, n) for x in props]
        
            for i in range(n):
                # 低层策略
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]

                # 高层策略 - 基础数据
                team[i].critic_map = all_critic_map_inp  # [num_processes, 3, H, W], 智能体共享一个全局图
                team[i].critic_nodes = all_critic_nodes  # [num_processes, num_agents, 4]
                team[i].goal = all_goals[i]
                team[i].task = all_tasks[i]
                team[i].higoal_log_prob = all_higoal_log_probs[i]
                team[i].high_value = all_high_value[i]  # [num_processes, 1]
                
                # 高层策略 - 图结构数据（用于PPO更新）
                team[i].ego_nodes = all_ego_nodes_split[i]  # [num_processes, 5]
                team[i].explore_nodes = all_explore_nodes_split[i]  # [num_processes, K, 4]
                team[i].landmark_data = all_landmark_datas_split[i]  # [num_processes, max_landmarks, 4]
                team[i].landmark_mask = all_landmark_masks_split_tensor[i]  # [num_processes, max_landmarks, 1]

                actions_list.append(all_action[i].cpu().numpy())
                goals_list.append(all_goals[i].cpu().numpy())
                tasks_list.append(all_tasks[i].cpu().numpy())

        return actions_list, goals_list, tasks_list

    def update(self):
        return_high_vals = []
        return_vals = []
        # use SMDP_ppo for training high level layer
        for i, trainer in enumerate(self.high_trainers_list):
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            high_vals = trainer.update(rollouts_list)
            return_high_vals.append([np.array(high_vals)]*len(rollouts_list))

        # use ippo ppo for training low level layer
        if self.use_pretrained_low_level:
            return_vals = [[np.array([0.0,0.0,0.0])]*len(self.teams_list[i]) for i in range(len(self.teams_list))]
        else:
            for i, trainer in enumerate(self.trainers_list):
                rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
                vals = trainer.update(rollouts_list)
                return_vals.append([np.array(vals)]*len(rollouts_list))

        low_arr = np.stack([x for v in return_vals for x in v])     # [num_agents, 3]
        high_arr = np.stack([x for v in return_high_vals for x in v])   # [num_agents, 3]

        # === 拼接 === [num_agents, 8]
        return np.concatenate([low_arr, high_arr], axis=1)
    
    def update_landmark_info(self,prev_landmark_data, prev_landmark_mask, detected_map_list, device, match_threshold=0.1, cleanup_threshold=0.06):
        """
        更新地标信息，结合之前的地标数据和当前检测到的地图信息。
        ⭐ 新增：自动清理与当前detected_map不一致的旧landmark
        
        参数:
        - prev_landmark_data: 上一步的地标数据，形状为 [num_agents * num_processes, max_landmarks, 4]
          每个 landmark 包含: [x, y, utility, is_targeted]
        - prev_landmark_mask: 上一步的地标掩码，形状为 [num_agents * num_processes, max_landmarks, 1]
          1 表示有效，0 表示空槽位
        - detected_map_list: list of torch.Tensor, 长度为 num_processes
          每个元素形状为 [num_detected, 2]，num_detected 可能不同
        - device: 设备（CPU 或 GPU）
        - match_threshold: landmark 匹配的距离阈值
        - cleanup_threshold: 清理旧landmark的距离阈值（如果landmark与所有detected点的距离都超过此值，则移除）
        
        返回:
        - updated_landmark_data: 更新后的地标数据，形状同 prev_landmark_data[num_agents * num_processes, max_landmarks, 4]
        - updated_landmark_mask: 更新后的地标掩码，形状同 prev_landmark_mask[num_agents * num_processes, max_landmarks, 1]
        """
        # 1. 深拷贝上一步的数据
        updated_landmark_data = prev_landmark_data.clone()
        updated_landmark_mask = prev_landmark_mask.clone()
        
        # 2. 解析形状信息
        num_agents_processes, max_landmarks, _ = prev_landmark_data.shape
        num_processes = len(detected_map_list)
        num_agents = num_agents_processes // num_processes
        
        
        # 4. 遍历每个 process，更新对应的 landmark 数据
        for proc_idx, detected_map in enumerate(detected_map_list):
            # detected_map: [num_detected, 2]，可能为空 [0, 2]
            num_detected = detected_map.shape[0]
            
            # 线性索引 = agent_idx * num_processes + proc_idx
            # 对于第一个智能体（agent_idx=0），线性索引 = proc_idx
            linear_idx = proc_idx  # 假设团队共享，使用第一个智能体的索引
            
            # 获取该智能体当前的 landmarks
            current_landmarks = updated_landmark_data[linear_idx]  # [max_landmarks, 4]
            current_mask = updated_landmark_mask[linear_idx]  # [max_landmarks, 1]
            
            # ========== 清理逻辑 ==========
            # 如果detected_map为空（例如环境reset后），清空所有landmark
            if num_detected == 0:
                updated_landmark_mask[linear_idx, :, 0] = 0.0
                updated_landmark_data[linear_idx, :, 3] = 0.0  # is_targeted也清0
                continue
            
            # 标记哪些旧landmark需要保留（即与detected_map中某个点距离<cleanup_threshold）
            valid_landmarks_indices = []
            for lm_idx in range(max_landmarks):
                if current_mask[lm_idx, 0] < 0.5:
                    continue  # 该槽位本身就是空的，跳过
                
                lm_pos = current_landmarks[lm_idx, :2]  # [2]
                # 计算到所有detected点的距离
                distances_to_detected = torch.norm(detected_map - lm_pos.unsqueeze(0), dim=1)  # [num_detected]
                min_dist = distances_to_detected.min()
                
                # 如果最近距离超过cleanup_threshold，说明这个landmark已经不存在了，移除
                if min_dist > cleanup_threshold:
                    updated_landmark_mask[linear_idx, lm_idx, 0] = 0.0
                    updated_landmark_data[linear_idx, lm_idx, 3] = 0.0
                else:
                    valid_landmarks_indices.append(lm_idx)
            
            # ========== 匹配或新增 landmark ==========
            for det_pos in detected_map:  # det_pos: [2]
                # 6.1 查找是否匹配现有 landmark
                matched_idx = self._find_landmark_match(det_pos, current_landmarks, 
                    current_mask, match_threshold)
                
                if matched_idx is not None:
                    # 6.2 更新已存在的 landmark位置为新旧位置的加权平均（更倾向新位置）
                    old_pos = updated_landmark_data[linear_idx, matched_idx, 0:2]
                    updated_landmark_data[linear_idx, matched_idx, 0:2] = 0.3 * old_pos + 0.7 * det_pos  # 70%新位置
                    # utility 和 is_targeted 保持不变，后续统一更新
                else:
                    # 6.3 寻找空闲槽位添加新 landmark
                    empty_idx = self._find_empty_slot(updated_landmark_mask[linear_idx])
                    
                    if empty_idx is not None:
                        updated_landmark_data[linear_idx, empty_idx, 0:2] = det_pos  # x, y
                        updated_landmark_data[linear_idx, empty_idx, 2] = 2.0  # utility
                        updated_landmark_data[linear_idx, empty_idx, 3] = 0.0  # is_targeted
                        updated_landmark_mask[linear_idx, empty_idx, 0] = 1.0  # 激活该槽位
                    else:
                        # 没有空闲槽位，跳过（可以打印警告）
                        print(f"Warning: No empty slot for new landmark at process {proc_idx}")

            # ==================== 7. 广播给该 process 的所有智能体 ====================
            # 将更新后的 landmark 数据复制给该 process 的其他智能体
            for agent_idx in range(1, num_agents):
                broadcast_linear_idx = agent_idx * num_processes + proc_idx
                updated_landmark_data[broadcast_linear_idx] = updated_landmark_data[linear_idx].clone()
                updated_landmark_mask[broadcast_linear_idx] = updated_landmark_mask[linear_idx].clone()
        
        return updated_landmark_data.to(device), updated_landmark_mask.to(device)

    def _find_landmark_match(self, position, landmarks_data, landmarks_mask, threshold):
        """
        在张量中查找匹配的 landmark
        
        参数:
        - position: [2] 检测到的位置
        - landmarks_data: [max_landmarks, 4] landmark 数据
        - landmarks_mask: [max_landmarks, 1] 有效性掩码
        - threshold: 匹配阈值
        
        返回:
        - matched_idx: int or None
        """
        # 只考虑有效的 landmarks
        valid_mask = landmarks_mask[:, 0] > 0.5  # [max_landmarks]
        
        if not valid_mask.any():
            return None
        
        # 计算距离
        landmark_positions = landmarks_data[:, 0:2]  # [max_landmarks, 2]
        distances = torch.norm(landmark_positions - position.unsqueeze(0), dim=1)  # [max_landmarks]
        
        # 对无效的 landmark 设置为无穷大
        distances = distances.masked_fill(~valid_mask, float('inf'))
        
        # 找到最近的 landmark
        min_dist, min_idx = distances.min(dim=0)
        
        if min_dist < threshold:
            return min_idx.item()
        
        return None

    def _find_empty_slot(self, landmarks_mask):
        """
        查找第一个空闲槽位
        
        参数:
        - landmarks_mask: [max_landmarks, 1] 有效性掩码
        
        返回:
        - empty_idx: int or None
        """
        empty_mask = landmarks_mask[:, 0] < 0.5  # [max_landmarks]
        
        if empty_mask.any():
            return empty_mask.nonzero(as_tuple=False)[0].item()
        
        return None
    
    def wrap_horizon(self):
        # 需要根据最后一步的obs计算next_value，然后传入每个agent的rollout中，因为目的是计算GAE，而GAE的每一步return都需要用到下一步的value
        # 因此需要计算128步的goals，也只需要新goals
        for team, policy in zip(self.teams_list,self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_env_state = torch.cat([agent.rollouts.env_states[-1] for agent in team])

            # 默认采取之前的目标分配，[num_agents * num_processes, 2]
            last_goals = torch.cat([agent.rollouts.goals[-1] for agent in team])

            # 提取上一步的 发现landamrk 数据 和 掩码
            all_landmark_datas = torch.cat([agent.rollouts.landmark_datas[-1] for agent in team])
            all_landmark_masks = torch.cat([agent.rollouts.landmark_masks[-1] for agent in team])

            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # 1. 批量构建 critic map input 和 critic vec input
            entropy_maps = torch.stack([torch.from_numpy(np.array(info['entropy_map'])).float() 
                                        for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            heatmaps = torch.stack([torch.from_numpy(np.array(info['heatmap'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            landmark_heatmaps = torch.stack([torch.from_numpy(np.array(info['landmark_heatmap'])).float() 
                                            for info in self.envs_info]).to(self.device)  # [num_processes, H, W]

            all_critic_map_inp = torch.stack([entropy_maps, heatmaps, landmark_heatmaps], dim=1)  # [num_processes, 3, H, W]

            # 2. 收集所有 goal_done 状态并构建mask
            goal_done_list = [info['goal_done'] for info in self.envs_info]  # list of lists
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)  # [num_processes, num_agents]
            # 收集当前的 world_step 信息，并广播为 [num_processes, num_agents, 1] 的张量
            agent_world_steps = torch.tensor(
                [info['world_steps'] for info in self.envs_info], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(1).repeat(1, num_agents).unsqueeze(-1)  # [num_processes, num_agents, 1]
            # 根据 world_step 生成 battery 信息: battery = (50 - step) / 50
            agent_batterys = (50.0 - agent_world_steps) / 50.0  # [num_processes, num_agents, 1]

            # 3. 批量生成 landmark node
            detected_maps = [torch.from_numpy(np.array(info['map'][1])).float().to(self.device) 
                            for info in self.envs_info] # [num_processes, x, 2], 不规则形状，每个 process 检测到的目标数量不同
            
            new_detected, new_detected_masks = self.update_landmark_info(all_landmark_datas, all_landmark_masks, detected_maps, self.device) 
            # [num_agents * num_processes, max_landmarks, 4], [num_agents * num_processes, max_landmarks, 1]

            # 4. 批量通过RTT生成K个候选目标点
            ## 生成两张地图
            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)  # [num_processes, num_agents, H, W]
            voronoi_masks = torch.stack([
                torch.stack([torch.from_numpy(np.array(info['voronoi_masks'][a])).float() 
                            for a in range(num_agents)]) 
                for info in self.envs_info
            ]).to(self.device)  # [num_processes, num_agents, H, W]   

            ## 从all_obs批量生成智能体信息（栅格索引） [num_processes, num_agents, 2]
            agent_positions = last_obs[:, 2:4].view(num_agents, num_processes, 2).transpose(0, 1) # 位置[num_processes, num_agents, 2]
            agent_vels = last_obs[:, 0:2].view(num_agents, num_processes, 2).transpose(0, 1)      # 速度[num_processes, num_agents, 2]
            agent_goals = last_goals.view(num_agents, num_processes, 2).transpose(0, 1) # [num_processes, num_agents, 2]
            ## 拼接成[num_processes, num_agents, 4]
            agent_nodes = torch.cat([agent_positions, agent_goals], dim=-1)  # [num_processes, num_agents, 4]
            ego_nodes = torch.cat([agent_positions, agent_vels, agent_batterys], dim=-1)  # [num_processes, num_agents, 5]

            if goal_done_mask.any():
                # 获取需要更新的索引 (process_idx, agent_idx)
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)  # [N, 2] where N is number of True values
                
                # 选择需要更新的智能体所在环境输入
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_entropy_maps[proc_indices],
                    voronoi_masks[proc_indices]
                ], dim=0)  # [2, N, num_agents, H, W]
                
                vec_inp_agents = agent_nodes[proc_indices]  # [N, num_agents, 4]

                # 4.2 通过RTT生成候选探索点
                batch_explore_nodes = policy.get_explore_nodes(self.top_k, self.rrt_max_iter, vec_inp_agents, map_inps, agent_indices)  # [B_pro, B_agents, K, 4]
                batch_explore_nodes = batch_explore_nodes.reshape(-1, batch_explore_nodes.shape[-2], batch_explore_nodes.shape[-1])  # [B_pro*B_agents, K, 4]
                # 4.3 ego nodes
                batch_ego_nodes = ego_nodes[proc_indices, agent_indices]  # [N, 5]
                # 4.4 landmark nodes
                linear_indices = agent_indices * num_processes + proc_indices
                batch_landmark_nodes, batch_landmark_node_masks = policy.get_landmark_nodes(
                    last_obs[:,2:4],  # [num_agents * num_processes, 2]
                    new_detected,                   # [num_agents * num_processes, max_landmarks, 4]
                    new_detected_masks,             # [num_agents * num_processes, max_landmarks, 1]
                    linear_indices                  # [N]
                )  # List of [L_i, 4], List of [L_i, 1]

                # 4.5 edge features
                batch_ego_to_explore_edges, batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks = policy.get_edge_features(
                    batch_explore_nodes,    # batch_ego_to_explore_edges: List of [K, 3], 长度为 N
                    batch_landmark_nodes,   # batch_ego_to_landmark_edges: List of [L_i, 3], 长度为 N
                    batch_landmark_node_masks   # batch_ego_to_landmark_edge_masks: List of [L_i, 1], 长度为 N
                )
                
                # 5. 批量执行高层策略
                batch_goals = policy.get_high_level_goal(
                    batch_ego_nodes, # Tensor shape [N, 5]
                    batch_explore_nodes, # Tensor shape [N, K, 4]
                    batch_ego_to_explore_edges, # List of Tensor shape [K, 3], length N
                    batch_landmark_nodes, # List of Tensor shape [L_i, 4], length N
                    batch_landmark_node_masks, # List of Tensor shape [L_i, 1], length N
                    batch_ego_to_landmark_edges, # List of Tensor shape [L_i, 3], length N
                    batch_ego_to_landmark_edge_masks # List of Tensor shape [L_i, 1], length N
                )

                # 6.2 更新目标和任务
                last_goals[linear_indices] = batch_goals["waypoints"]  # [N, 2]
            
            all_critic_nodes = torch.cat([last_obs[:,2:4].view(num_agents, num_processes, -1).transpose(0, 1), 
                                          last_goals.view(num_agents, num_processes, 2).transpose(0, 1)], dim=-1)
            with torch.no_grad():
                next_high_value = policy.get_high_value(all_critic_map_inp, all_critic_nodes) # 计算所有process的高层value： [num_processes, num_agents]
                next_low_value = policy.get_low_value(last_obs, last_goals)

            all_high_value = torch.chunk(next_high_value,len(team), dim=1)
            all_low_value = torch.chunk(next_low_value,len(team))
            for i in range(len(team)):
                team[i].wrap_horizon(all_low_value[i], all_high_value[i])

    def after_update(self):
        for agent in self.all_agents:
            agent.after_update()
    
    def initial_hidden_states(self, step):
        for agent in self.all_agents:
            agent.initial_hidden_states(step)

    def update_rollout(self, obs, reward, high_rewards, masks, env_state, goal_dones):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        env_state_t = torch.from_numpy(env_state).float().to(self.device)
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t[:, i, :]
            agent.update_rollout(agent_obs, reward[:,i].unsqueeze(1), high_rewards[:,i].unsqueeze(1), 
                                 masks[:,i].unsqueeze(1), env_state_t, goal_dones[:,i].unsqueeze(1))

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, env_states, goals, tasks, landmark_data, landmark_mask, deterministic=True):
        # used only while evaluating policies. Assuming that agents are in order of team!
        # goals: 上一步的目标分配 [num_agents, 2]
        # landmark_data: 上一步的地标数据 [num_agents, max_landmarks, 4]
        # landmark_mask: 上一步的地标掩码 [num_agents, max_landmarks, 1]
        obs1 = []
        obs2 = []
        all_obs = []

        graph_data = {
            'ego_nodes': [],           # 自我节点位置 [num_agents, 2]
            'explore_nodes': [],       # 探索节点 [num_agents, K, 2]
            'landmark_nodes': [],      # landmark节点 [num_agents, L, 2]
            'ego_to_explore_edges': [], # 自我到探索节点的边
            'ego_to_landmark_edges': [], # 自我到landmark的边
            'selected_goal_idx': [],   # 选中的目标索引
            'selected_task': []        # 选中的任务类型
        }

        for i in range(len(obs)):
            agent = self.env.world.policy_agents[i]
            if hasattr(agent, 'adversary') and agent.adversary:
                obs1.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
            else:
                obs2.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        if len(obs1)!=0:
            all_obs.append(obs1)
        if len(obs2)!=0:
            all_obs.append(obs2)

        actions = []
        # 这里需要对env_states进行处理，因为它是(env_state_dim)的形状，需要复制成(num_agent, env_state_dim)
        env_states = torch.from_numpy(env_states).float().to(self.device)
        for team,policy,obs in zip(self.teams_list,self.policies_list,all_obs):
            # 默认采取之前的目标分配，[num_agents, 2]
            all_goals = goals
            all_tasks = tasks

            num_agents = len(team)

            # 1. 收集数据

            obs_tensor = torch.cat(obs, dim=0).to(self.device) # [num_agents, obs_dim]

            # 1.1. 构建 entropy_map, heatmap, landmark_heatmap
            entropy_map = torch.from_numpy(np.array(self.envs_info['entropy_map'])).float().unsqueeze(0).to(self.device)  # [1, H, W]
            heatmap = torch.from_numpy(np.array(self.envs_info['heatmap'])).float().unsqueeze(0).to(self.device)  # [1, H, W]
            landmark_heatmap = torch.from_numpy(np.array(self.envs_info['landmark_heatmap'])).float().unsqueeze(0).to(self.device)  # [1, H, W]

            # 1.2. 收集所有 goal_done 状态并构建mask
            goal_done_list = [self.envs_info['goal_done']]  # ✅ 包装成列表
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)  # [1, num_agents]   

            # 1.3. 收集当前的 world_step 信息，并广播为 [1, num_agents, 1] 的张量
            agent_world_steps = torch.tensor(
                [self.envs_info['world_steps']], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(1).repeat(1, num_agents).unsqueeze(-1)  # [1, num_agents, 1]
            # 根据 world_step 生成 battery 信息: battery = (50 - step) / 50
            agent_batterys = (50.0 - agent_world_steps) / 50.0  # [1, num_agents, 1]

            # 1.4. 收集 landmark data 和 mask
            detected_map = torch.from_numpy(np.array(self.envs_info['map'][1])).float().to(self.device)

            new_detected, new_detected_masks = self.update_landmark_info(
                landmark_data, 
                landmark_mask, 
                [detected_map], 
                self.device
            )  # [num_agents * 1, max_landmarks, 4], [num_agents * 1, max_landmarks, 1]

            # 2. 生成动态异构图结构的节点表示
            # 2.1. 输入准备，包括地图输入和向量输入
            agent_entropy_map = entropy_map.unsqueeze(1).repeat(1, num_agents, 1, 1) # [num_processes, num_agents, H, W]
            voronoi_masks = torch.stack([
                torch.from_numpy(np.array(self.envs_info['voronoi_masks'][a])).float()
                for a in range(num_agents)
            ]).unsqueeze(0).to(self.device)  # [1, num_agents, H, W]

            ## 从all_obs批量生成智能体信息（栅格索引） [num_processes, num_agents, 2]
            agent_positions = obs_tensor[:, 2:4].view(1, num_agents, 2) # 位置[1, num_agents, 2]
            agent_vels = obs_tensor[:, 0:2].view(1, num_agents, 2)      # 速度[1, num_agents, 2]
            agent_goals = all_goals.view(1, num_agents, 2) # [1, num_agents, 2]
            ## 拼接成[1, num_agents, 4]
            agent_nodes = torch.cat([agent_positions, agent_goals], dim=-1)  # [1, num_agents, 4]
            ego_nodes = torch.cat([agent_positions, agent_vels, agent_batterys], dim=-1)  # [1, num_agents, 5]

            if goal_done_mask.any():
                # 获取需要更新的索引 (process_idx, agent_idx)
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)  # [N, 2] where N is number of True values
                   
                # 只选择需要更新的智能体
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_entropy_map[proc_indices],
                    voronoi_masks[proc_indices],
                ], dim=0)  # [2, N, num_agents, H, W]
                
                vec_inp_agents = agent_nodes[proc_indices]  # [N, num_agents, 4]

                # 2.2. 通过RTT生成候选探索点
                batch_explore_nodes = policy.get_explore_nodes(self.top_k, self.rrt_max_iter, vec_inp_agents, map_inps, agent_indices)  # [B_pro, B_agents, K, 4]
                batch_explore_nodes = batch_explore_nodes.reshape(-1, batch_explore_nodes.shape[-2], batch_explore_nodes.shape[-1])  # [B_pro*B_agents, K, 4]
                print("batch_explore_nodes:", batch_explore_nodes)
                # 2.3. ego nodes
                batch_ego_nodes = ego_nodes[proc_indices, agent_indices]  # [N, 5]
                # 2.4. landmark nodes
                linear_indices = agent_indices * 1 + proc_indices
                batch_landmark_nodes, batch_landmark_node_masks = policy.get_landmark_nodes(
                    obs_tensor[:,2:4],  # [num_agents * 1, 2]
                    new_detected,                   # [num_agents * 1, max_landmarks, 4]
                    new_detected_masks,             # [num_agents * 1, max_landmarks, 1]
                    linear_indices                  # [N]
                )  # List of [L_i, 4], List of [L_i, 1]

                # 2.5. edge features
                batch_ego_to_explore_edges, batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks = policy.get_edge_features(
                    batch_explore_nodes,    # batch_ego_to_explore_edges: List of [K, 3], 长度为 N
                    batch_landmark_nodes,   # batch_ego_to_landmark_edges: List of [L_i, 3], 长度为 N
                    batch_landmark_node_masks   # batch_ego_to_landmark_edge_masks: List of [L_i, 1], 长度为 N
                )

                # 3. 批量执行高层策略
                batch_goals = policy.get_high_level_goal(
                    batch_ego_nodes, # Tensor shape [N, 5]
                    batch_explore_nodes, # Tensor shape [N, K, 4]
                    batch_ego_to_explore_edges, # Tensor [N, K, 3]
                    batch_landmark_nodes, # Tensor: [N, Max_L, 4]
                    batch_landmark_node_masks, # Tensor: [N, Max_L, 1]
                    batch_ego_to_landmark_edges, # Tensor [N, Max_L, 3]
                    batch_ego_to_landmark_edge_masks, # Tensor [N, Max_L, 1]
                    deterministic  = deterministic
                    )  # 需要实现batch版本
                
                # 4. 批量更新 all_goals 和 all_tasks
                # 4.1.计算线性索引: agent_idx * num_processes + process_idx
                linear_indices = agent_indices * 1 + proc_indices
                
                # 4.2. 更新目标和任务
                all_goals[linear_indices] = batch_goals["waypoints"]  # [N, 2] 转换为float
                all_tasks[linear_indices] = batch_goals["action_modes"]  # [N, 1] 转换为float

                # 4.3. 更新 landmark data 和 mask
                # 如果智能体选择的任务是 landmark，则更新对应 landmark 的 is_targeted 属性
                for i, lin_idx in enumerate(linear_indices):
                    if batch_goals["action_modes"][i, 0] == 1:  # 选择的是 landmark
                        # 获取选中的 waypoint（绝对世界坐标）
                        selected_waypoint = batch_goals["waypoints"][i]  # [2]
                        
                        # 在对应的 landmark 列表中找到匹配的 landmark
                        landmarks = new_detected[lin_idx]  # [max_landmarks, 4]
                        landmark_mask = new_detected_masks[lin_idx]  # [max_landmarks, 1]
                        
                        # 找到有效的 landmark
                        valid_mask = landmark_mask[:, 0] > 0.5
                        if valid_mask.any():
                            # 计算距离（使用绝对坐标）
                            landmark_positions = landmarks[:, :2]  # [max_landmarks, 2]
                            distances = torch.norm(landmark_positions - selected_waypoint, dim=1)  # [max_landmarks]
                            distances = distances.masked_fill(~valid_mask, float('inf'))
                            
                            # 找到最近的 landmark
                            min_idx = distances.argmin()
                            if distances[min_idx] < 0.05:  # 匹配阈值 0.05
                                # new_detected[lin_idx, min_idx, 3] = 1.0  # 设置 is_targeted = 1
                                # 所有agent的障碍物is_targeted同步更新
                                new_detected[:, min_idx, 3] = 1.0
                    
                    # 自我节点（世界坐标）
                    ego_pos = batch_ego_nodes[i, :2].cpu().numpy()
                    graph_data['ego_nodes'].append(ego_pos)
                    print("ego_pos:", ego_pos)
                    
                    # 探索节点（转换为世界坐标）
                    explore_nodes_local = batch_explore_nodes[i].cpu().numpy()  # [K, 4], 最后2维是相对坐标
                    explore_nodes_world = explore_nodes_local[:, :2] + ego_pos  # 转换为世界坐标
                    print("explore_nodes_world:", explore_nodes_world)
                    graph_data['explore_nodes'].append(explore_nodes_world)
                    
                    # Landmark节点（已经是世界坐标）
                    landmark_nodes_local = batch_landmark_nodes[i][:, :2].cpu().numpy()  # [L, 2]
                    landmark_nodes_world = landmark_nodes_local + ego_pos  # 转换为世界坐标
                    print("landmark_nodes_world:", landmark_nodes_world)
                    graph_data['landmark_nodes'].append(landmark_nodes_world)
                    
                    # 边的信息（距离、角度等）
                    graph_data['ego_to_explore_edges'].append(
                        batch_ego_to_explore_edges[i].cpu().numpy()
                    )
                    graph_data['ego_to_landmark_edges'].append(
                        batch_ego_to_landmark_edges[i].cpu().numpy()
                    )
                    # 选中的目标和任务
                    selected_task = batch_goals["action_modes"][i, 0].item()
                    graph_data['selected_task'].append(selected_task)
                    
                    # 根据任务类型确定选中的节点索引
                    if selected_task == 0:  # 探索节点
                        selected_waypoint = batch_goals["waypoints"][i].cpu().numpy()
                        distances = np.linalg.norm(explore_nodes_world - selected_waypoint, axis=1)
                        selected_idx = int(np.argmin(distances))
                    else:  # landmark节点
                        selected_waypoint = batch_goals["waypoints"][i].cpu().numpy()
                        distances = np.linalg.norm(landmark_nodes_world - selected_waypoint, axis=1)
                        selected_idx = int(np.argmin(distances))
                    
                    graph_data['selected_goal_idx'].append(selected_idx)
                
                # 更新所有智能体的 landmark_data 和 landmark_mask
                landmark_data = new_detected
                landmark_mask = new_detected_masks

            if len(obs)!=0:
                _,action,_ = policy.low_level_act(obs_tensor, all_goals, deterministic=True)
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions), all_goals, all_tasks, landmark_data, landmark_mask, graph_data

    def eval_reward_choose(self, all_rewards, task):
        n,dim = task.shape
        task_reshape = task.transpose(0,1).reshape(n*dim)
        masked = all_rewards * task_reshape
        agent_reward = masked.view(dim, n).sum(dim=0)
        return agent_reward

    def set_eval_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.train()
