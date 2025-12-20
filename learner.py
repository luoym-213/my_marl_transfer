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
            all_map_log_probs = torch.cat([agent.rollouts.map_log_probs[step-1] for agent in team])
            all_decision_log_probs = torch.cat([agent.rollouts.decision_log_probs[step-1] for agent in team])

            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # 1. 批量构建 critic map input 和 critic vec input
            belief_maps = torch.stack([torch.from_numpy(np.array(info['belief_map'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            entropy_maps = torch.stack([torch.from_numpy(np.array(info['entropy_map'])).float() 
                                        for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            heatmaps = torch.stack([torch.from_numpy(np.array(info['heatmap'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            landmark_heatmaps = torch.stack([torch.from_numpy(np.array(info['landmark_heatmap'])).float() 
                                            for info in self.envs_info]).to(self.device)  # [num_processes, H, W]

            all_critic_map_inp = torch.stack([belief_maps, entropy_maps, heatmaps, landmark_heatmaps], dim=1)  # [num_processes, 4, H, W]
            all_critic_vec_inp = all_env_state[:len(self.envs_info),6:] # 向量流：[num_processes, num_agents*2 + num_landmarks*2]

            # 2. 收集所有 goal_done 状态并构建mask
            goal_done_list = [info['goal_done'] for info in self.envs_info]  # list of lists
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)  # [num_processes, num_agents]

            # 3. 批量生成 vec_inp
            detected_maps = [torch.from_numpy(np.array(info['map'][1])).float().to(self.device) 
                            for info in self.envs_info]
            vec_inps = torch.stack([policy.vec_inp_generator(env_state, detected_map) 
                                    for env_state, detected_map in zip(all_env_state[:num_processes], detected_maps)])  
            # [num_processes, num_agents, 5]

            # 4. 批量准备需要更新的智能体数据                
            # 批量构建 map_inp for all agents that need update
            agent_belief_maps = belief_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)  # [num_processes, num_agents, H, W]
            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)

            voronoi_masks = torch.stack([
                torch.stack([torch.from_numpy(np.array(info['voronoi_masks'][a])).float() 
                            for a in range(num_agents)]) 
                for info in self.envs_info
            ]).to(self.device)  # [num_processes, num_agents, H, W]
            
            distance_fields = torch.stack([
                torch.stack([torch.from_numpy(np.array(info['distance_fields'][a])).float() 
                            for a in range(num_agents)]) 
                for info in self.envs_info
            ]).to(self.device)  # [num_processes, num_agents, H, W]

            if goal_done_mask.any():
                # 获取需要更新的索引 (process_idx, agent_idx)
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)  # [N, 2] where N is number of True values                
                
                # 只选择需要更新的智能体
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_belief_maps[proc_indices, agent_indices],
                    agent_entropy_maps[proc_indices, agent_indices],
                    voronoi_masks[proc_indices, agent_indices],
                    distance_fields[proc_indices, agent_indices]
                ], dim=1)  # [N, 4, H, W]
                
                vec_inp_agents = vec_inps[proc_indices, agent_indices]  # [N, 5]
                
                # 5. 批量执行高层策略
                # batch_goals = policy.get_high_level_goal(vec_inp_agents, map_inps)  # 需要实现batch版本
                batch_goals = policy.get_random_high_level_goal(vec_inp_agents, map_inps)  # 需要实现batch版本
                
                # 6. 批量更新 all_goals 和 all_tasks
                # 计算线性索引: agent_idx * num_processes + process_idx
                linear_indices = agent_indices * num_processes + proc_indices
                
                all_goals[linear_indices] = batch_goals["waypoints"]  # [N, 2]
                all_tasks[linear_indices] = batch_goals["action_modes"]  # [N, 1]
                all_map_log_probs[linear_indices] = batch_goals["map_log_probs"]  # [N, 1]
                all_decision_log_probs[linear_indices] = batch_goals["decision_log_probs"]  # [N, 1]

            # 7. 保存全量的map_inps和vec_inps到rollouts中，以便后续evaluate使用
            all_map_inps = torch.stack([
                agent_belief_maps, # [num_processes, num_agents, H, W]
                agent_entropy_maps, # [num_processes, num_agents, H, W]
                voronoi_masks,      # [num_processes, num_agents, H, W]
                distance_fields     # [num_processes, num_agents, H, W]
            ], dim=2)  # [num_processes, num_agents, 4, H, W]
            
            all_vec_inps = vec_inps  # [num_processes, num_agents, 5]

            #all_high_value = policy.get_high_value(all_critic_map_inp, all_critic_vec_inp) # 计算所有process的高层value： [num_processes, num_agents]
            #随机产生一个all_high_value以避免影响训练
            all_high_value = torch.zeros((num_processes, num_agents), device=self.device)

            # 低层策略的处理
            # props = policy.act(all_obs, all_hidden, all_env_state, all_masks, deterministic=False) # a single forward pass
            props = policy.low_level_act(all_obs, all_goals, deterministic=False) # a single forward pass

            # split all outputs
            n = len(team)
            all_map_inps = torch.chunk(all_map_inps, n, dim=1)  # 按智能体拆分
            all_vec_inps = torch.chunk(all_vec_inps, n, dim=1)
            all_goals = torch.chunk(all_goals, n)
            all_tasks = torch.chunk(all_tasks, n)
            all_map_log_probs = torch.chunk(all_map_log_probs, n)
            all_decision_log_probs = torch.chunk(all_decision_log_probs, n)
            all_high_value = torch.chunk(all_high_value, n, dim=1)


            all_value, all_action, all_action_log_prob = [torch.chunk(x, n) for x in props]
            for i in range(n):
                # 低层策略
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]

                # 高层策略
                team[i].map_obs = all_map_inps[i].squeeze(1)  # [num_processes, 4, H, W]
                team[i].vec_obs = all_vec_inps[i].squeeze(1)  # [num_processes, vec_dim]
                team[i].critic_map = all_critic_map_inp # [num_processes, 4, H, W], 智能体共享一个全局图
                team[i].goal = all_goals[i]
                team[i].task = all_tasks[i]
                team[i].map_log_prob = all_map_log_probs[i]
                team[i].decision_log_prob = all_decision_log_probs[i]
                team[i].high_value = all_high_value[i]  # [num_processes, 1]

                actions_list.append(all_action[i].cpu().numpy())
                goals_list.append(all_goals[i].cpu().numpy())

        return actions_list, goals_list

    def update(self):
        return_high_vals = []
        return_vals = []
        # use SMDP_ppo for training high level layer
        # for i, trainer in enumerate(self.high_trainers_list):
        #     rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
        #     high_vals = trainer.update(rollouts_list)
        #     return_high_vals.append([np.array(high_vals)]*len(rollouts_list))
        return_high_vals = [[np.array([0.0,0.0,0.0,0.0,0.0])]*len(self.teams_list[i]) for i in range(len(self.teams_list))]

        # use ippo ppo for training low level layer
        if self.use_pretrained_low_level:
            return_vals = [[np.array([0.0,0.0,0.0])]*len(self.teams_list[i]) for i in range(len(self.teams_list))]
        else:
            for i, trainer in enumerate(self.trainers_list):
                rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
                vals = trainer.update(rollouts_list)
                return_vals.append([np.array(vals)]*len(rollouts_list))

        low_arr = np.stack([x for v in return_vals for x in v])     # [num_agents, 3]
        high_arr = np.stack([x for v in return_high_vals for x in v])   # [num_agents, 5]

        # === 拼接 === [num_agents, 8]
        return np.concatenate([low_arr, high_arr], axis=1)

    def wrap_horizon(self):
        # 需要根据最后一步的obs计算next_value，然后传入每个agent的rollout中，因为目的是计算GAE，而GAE的每一步return都需要用到下一步的value
        # 因此需要计算128步的goals，也只需要新goals
        for team, policy in zip(self.teams_list,self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_env_state = torch.cat([agent.rollouts.env_states[-1] for agent in team])

            # 默认采取之前的目标分配，[num_agents * num_processes, 2]
            last_goals = torch.cat([agent.rollouts.goals[-1] for agent in team])


            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # 1. 批量构建 critic map input 和 critic vec input
            belief_maps = torch.stack([torch.from_numpy(np.array(info['belief_map'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            entropy_maps = torch.stack([torch.from_numpy(np.array(info['entropy_map'])).float() 
                                        for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            heatmaps = torch.stack([torch.from_numpy(np.array(info['heatmap'])).float() 
                                    for info in self.envs_info]).to(self.device)  # [num_processes, H, W]
            landmark_heatmaps = torch.stack([torch.from_numpy(np.array(info['landmark_heatmap'])).float() 
                                            for info in self.envs_info]).to(self.device)  # [num_processes, H, W]

            all_critic_map_inp = torch.stack([belief_maps, entropy_maps, heatmaps, landmark_heatmaps], dim=1)  # [num_processes, 4, H, W]
            all_critic_vec_inp = last_env_state[:len(self.envs_info),2*num_agents:] # 向量流：[num_processes, num_agents*2 + num_landmarks*2]

            # 2. 收集所有 goal_done 状态并构建mask
            goal_done_list = [info['goal_done'] for info in self.envs_info]  # list of lists
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)  # [num_processes, num_agents]

            # 3. 批量生成 vec_inp
            detected_maps = [torch.from_numpy(np.array(info['map'][1])).float().to(self.device) 
                            for info in self.envs_info]
            vec_inps = torch.stack([policy.vec_inp_generator(env_state, detected_map) 
                                    for env_state, detected_map in zip(last_env_state[:num_processes], detected_maps)])  
            # [num_processes, num_agents, 5]

            # 4. 批量准备需要更新的智能体数据
            if goal_done_mask.any():
                # 获取需要更新的索引 (process_idx, agent_idx)
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)  # [N, 2] where N is number of True values
                
                # 批量构建 map_inp for all agents that need update
                agent_belief_maps = belief_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)  # [num_processes, num_agents, H, W]
                agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)
                
                voronoi_masks = torch.stack([
                    torch.stack([torch.from_numpy(np.array(info['voronoi_masks'][a])).float() 
                                for a in range(num_agents)]) 
                    for info in self.envs_info
                ]).to(self.device)  # [num_processes, num_agents, H, W]
                
                distance_fields = torch.stack([
                    torch.stack([torch.from_numpy(np.array(info['distance_fields'][a])).float() 
                                for a in range(num_agents)]) 
                    for info in self.envs_info
                ]).to(self.device)  # [num_processes, num_agents, H, W]
                
                # 只选择需要更新的智能体
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_belief_maps[proc_indices, agent_indices],
                    agent_entropy_maps[proc_indices, agent_indices],
                    voronoi_masks[proc_indices, agent_indices],
                    distance_fields[proc_indices, agent_indices]
                ], dim=1)  # [N, 4, H, W]
                
                vec_inp_agents = vec_inps[proc_indices, agent_indices]  # [N, 5]
                
                # 5. 批量执行高层策略
                batch_goals = policy.get_high_level_goal(vec_inp_agents, map_inps)  # 需要实现batch版本
                
                # 6. 批量更新 all_goals 和 all_tasks
                # 计算线性索引: agent_idx * num_processes + process_idx
                linear_indices = agent_indices * num_processes + proc_indices
                last_goals[linear_indices] = batch_goals["waypoints"]  # [N, 2]
            
            with torch.no_grad():
                next_high_value = policy.get_high_value(all_critic_map_inp, all_critic_vec_inp) # 计算所有process的高层value： [num_processes, num_agents]
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

    def eval_act(self, obs, env_states, goals):
        # used only while evaluating policies. Assuming that agents are in order of team!
        # goals: 上一步的目标分配 [num_agents, 2]
        obs1 = []
        obs2 = []
        all_obs = []
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

            num_agents = len(team)

            belief_map = torch.from_numpy(np.array(self.envs_info['belief_map'])).float().unsqueeze(0).to(self.device)  # [1, H, W]
            entropy_map = torch.from_numpy(np.array(self.envs_info['entropy_map'])).float().unsqueeze(0).to(self.device)  # [1, H, W]

            # 1. 收集所有 goal_done 状态并构建mask
            goal_done_list = [self.envs_info['goal_done']]  # ✅ 包装成列表
            goal_done_mask = torch.tensor(goal_done_list, dtype=torch.bool, device=self.device)  # [1, num_agents]   

            # 2. 生成vec_inp
            detected_map = torch.from_numpy(np.array(self.envs_info['map'][1])).float().to(self.device)
            vec_inp = policy.vec_inp_generator(env_states, detected_map).unsqueeze(0)  # [1, num_agents, 5]  
            # [1, num_agents, 5]

            # 批量构建 map_inp for all agents that need update
            agent_belief_maps = belief_map.unsqueeze(1).repeat(1, num_agents, 1, 1)  # [1, num_agents, H, W]
            agent_entropy_maps = entropy_map.unsqueeze(1).repeat(1, num_agents, 1, 1)
            
            voronoi_masks = torch.stack([
                torch.from_numpy(np.array(self.envs_info['voronoi_masks'][a])).float()
                for a in range(num_agents)
            ]).unsqueeze(0).to(self.device)  # [1, num_agents, H, W]
            
            distance_fields = torch.stack([
                torch.from_numpy(np.array(self.envs_info['distance_fields'][a])).float()
                for a in range(num_agents)
            ]).unsqueeze(0).to(self.device)  # [1, num_agents, H, W]

            # 3. 准备需要更新的智能体数据
            if goal_done_mask.any():
                # 获取需要更新的索引 (process_idx, agent_idx)
                update_indices = torch.nonzero(goal_done_mask, as_tuple=False)  # [N, 2] where N is number of True values
                   
                # 只选择需要更新的智能体
                proc_indices = update_indices[:, 0]
                agent_indices = update_indices[:, 1]
                
                map_inps = torch.stack([
                    agent_belief_maps[proc_indices, agent_indices],
                    agent_entropy_maps[proc_indices, agent_indices],
                    voronoi_masks[proc_indices, agent_indices],
                    distance_fields[proc_indices, agent_indices]
                ], dim=1)  # [N, 4, H, W]
                
                vec_inp_agents = vec_inp[proc_indices, agent_indices]  # [N, 5]

                # 5. 批量执行高层策略
                batch_goals = policy.get_high_level_goal(vec_inp_agents, map_inps)  # 需要实现batch版本
                
                # 6. 批量更新 all_goals 和 all_tasks
                # 计算线性索引: agent_idx * num_processes + process_idx
                linear_indices = agent_indices * 1 + proc_indices
                
                all_goals[linear_indices] = batch_goals["waypoints"]  # [N, 2]

            if len(obs)!=0:
                _,action,_ = policy.low_level_act(torch.cat(obs).to(self.device), all_goals, deterministic=True)
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions), all_goals

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
