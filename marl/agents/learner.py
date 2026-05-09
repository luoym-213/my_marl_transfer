import torch
from marl.algos import IPPO, JointPPO
from marl.agents.factory import setup_master as setup_master_from_factory
from marl.agents.updates import update_policies
from marl.controllers.eval_controller import EvalController
from marl.controllers.high_level_context import (
    agent_batteries,
    build_agent_context,
    detected_maps as build_detected_maps,
    goal_done_mask as build_goal_done_mask,
    stack_global_maps,
    voronoi_masks as build_voronoi_masks,
)
from marl.controllers.high_level_policy import HighLevelPolicy
from marl.controllers.landmark_memory import LandmarkMemory
from marl.controllers.rollout_writer import RolloutWriter


def setup_master(args, env=None, return_env=False):
    return setup_master_from_factory(
        args, Learner, env=env, return_env=return_env
    )


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
        self.landmark_memory = LandmarkMemory(self.device)
        self.high_level_policy = HighLevelPolicy(
            args.top_k, args.rrt_max_iter, self.device
        )
        self.rollout_writer = RolloutWriter(self.device)
        self.eval_controller = EvalController(self.high_level_policy, self.device)

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
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

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

            # 从all_masks中提取env_done信息，全0表示当前process的episode结束
            episode_dones = all_masks.view(num_agents, num_processes).transpose(0, 1)
            env_dones = (episode_dones.sum(dim=1) == 0)  # [num_processes]，True表示该process的episode结束

            # ⭐ 构建Critic输入
            entropy_maps, _, _, all_critic_map_inp = stack_global_maps(
                self.envs_info, self.device
            )
            goal_done_mask = build_goal_done_mask(self.envs_info, self.device)
            agent_batterys = agent_batteries(
                self.envs_info, num_agents, self.device
            )

            # ⭐ 生成landmark节点
            detected_maps = build_detected_maps(self.envs_info, self.device)
            new_detected, new_detected_masks = self.update_landmark_info(all_landmark_datas, 
                                                                         all_landmark_masks, 
                                                                         detected_maps, 
                                                                         self.device, 
                                                                         env_dones)
            # new_detected: Tensor shape [num_agents * num_processes, max_landmarks, 4], agent first
            # new_detected_masks: Tensor shape [num_agents * num_processes, max_landmarks, 1], agent first


            # ⭐ 准备智能体节点数据
            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)
            voronoi_masks_t = build_voronoi_masks(
                self.envs_info, num_agents, self.device
            )
            agent_context = build_agent_context(
                all_obs, all_goals, all_masks,
                agent_batterys, num_agents, num_processes
            )
            agent_positions = agent_context["agent_positions"]
            agent_nodes = agent_context["agent_nodes"]
            ego_nodes = agent_context["ego_nodes"]
            teammate_nodes = agent_context["teammate_nodes"]
            global_teammate_mask = agent_context["teammate_mask"]

            # landmark node docker: Tensor shape [num_agents * num_processes, Max_L, 4]
            all_landmark_nodes = torch.zeros(num_processes * num_agents, new_detected.shape[1], 4, device=self.device)

            high_decision = self.high_level_policy.select_goals(
                policy=policy,
                obs_positions=all_obs[:, 2:4],
                all_masks=all_masks,
                goals=all_goals,
                tasks=all_tasks,
                goal_log_probs=all_higoal_log_probs,
                goal_done_mask=goal_done_mask,
                agent_entropy_maps=agent_entropy_maps,
                voronoi_masks=voronoi_masks_t,
                agent_nodes=agent_nodes,
                ego_nodes=ego_nodes,
                teammate_nodes=teammate_nodes,
                teammate_mask=global_teammate_mask,
                landmark_data=new_detected,
                landmark_mask=new_detected_masks,
                num_processes=num_processes,
                deterministic=False,
                update_tasks=True,
                update_log_probs=True,
                update_targeted=True,
                store_landmark_nodes=True,
                landmark_nodes_out=all_landmark_nodes,
            )
            all_goals = high_decision["goals"]
            all_tasks = high_decision["tasks"]
            all_higoal_log_probs = high_decision["goal_log_probs"]
            all_landmark_datas = high_decision["landmark_data"]
            all_landmark_masks = high_decision["landmark_mask"]
            all_landmark_nodes = high_decision["landmark_nodes"]

            # ⭐ 准备rollout数据
            # 转换 ego_nodes 为 agent-major 顺序，以匹配后续的 chunk 操作
            # [P, A, D] -> [A, P, D] -> [A*P, D]
            all_ego_nodes = ego_nodes.transpose(0, 1).contiguous().view(num_processes * num_agents, -1)
            K = self.top_k
            # ⭐ 使用 repeat 确保环境索引在 chunk 后能正确分配给每个智能体
            all_teammate_nodes = teammate_nodes.repeat(num_agents, 1, 1)    # [num_agents * num_processes, num_agents, 5]
            all_teammate_masks = global_teammate_mask.repeat(num_agents, 1, 1)  # [num_agents * num_processes, num_agents, 1]
            all_explore_nodes = torch.zeros(num_processes * num_agents, K, 4, device=self.device)
            if high_decision["has_decision"]:
                # 将决策智能体的 explore nodes 填充到对应位置
                all_explore_nodes[high_decision["linear_indices"]] = high_decision["explore_nodes"]
                # 将teammate_masks中self-masking的部分也更新
                all_teammate_masks[high_decision["linear_indices"]] = high_decision["teammate_masks"]
                
            # 8. 计算高层value
            # Vector Input: Tensor shape [Batch, N_agents, 4] [x,y,x_g,y_g]
            # Map Input: Tensor shape [Batch, 3, H, W]。
            # all_critic_vec_inp = [x,y,x_g,y_g]
            ## 拼接成[num_processes, num_agents, 4]
            all_critic_nodes = torch.cat([agent_positions, all_goals.view(num_agents, num_processes, 2).transpose(0, 1)], dim=-1)  # [num_processes, num_agents, 4]
            all_high_value = policy.get_high_value(all_critic_map_inp, all_critic_nodes) # 计算所有process的高层value： [num_processes, num_agents]

            # ⭐ 底层策略
            props = policy.low_level_act(all_obs, all_goals, deterministic=False)

            team_actions, team_goals, team_tasks = self.rollout_writer.assign_act_outputs(
                team=team,
                low_outputs=props,
                goals=all_goals,
                tasks=all_tasks,
                goal_log_probs=all_higoal_log_probs,
                high_values=all_high_value,
                critic_map=all_critic_map_inp,
                critic_nodes=all_critic_nodes,
                ego_nodes=all_ego_nodes,
                explore_nodes=all_explore_nodes,
                teammate_nodes=all_teammate_nodes,
                teammate_masks=all_teammate_masks,
                landmark_data=all_landmark_datas,
                landmark_mask=all_landmark_masks,
                landmark_nodes=all_landmark_nodes,
            )
            actions_list.extend(team_actions)
            goals_list.extend(team_goals)
            tasks_list.extend(team_tasks)

        return actions_list, goals_list, tasks_list

    def update(self):
        trainers_list = None if self.use_pretrained_low_level else self.trainers_list
        return update_policies(
            self.teams_list,
            trainers_list,
            self.high_trainers_list,
            self.use_pretrained_low_level,
        )
    
    def update_landmark_info(self, prev_landmark_data, prev_landmark_mask,
                             detected_map_list, device, env_dones=None,
                             match_threshold=0.1, cleanup_threshold=0.06):
        return self.landmark_memory.update(
            prev_landmark_data,
            prev_landmark_mask,
            detected_map_list,
            device=device,
            env_dones=env_dones,
            match_threshold=match_threshold,
            cleanup_threshold=cleanup_threshold,
        )
    
    def wrap_horizon(self):
        # 需要根据最后一步的obs计算next_value，然后传入每个agent的rollout中，因为目的是计算GAE，而GAE的每一步return都需要用到下一步的value
        # 因此需要计算128步的goals，也只需要新goals
        for team, policy in zip(self.teams_list,self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])

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

            # 从all_masks中提取env_done信息，全0表示当前process的episode结束
            episode_dones = last_masks.view(num_agents, num_processes).transpose(0, 1)
            env_dones = (episode_dones.sum(dim=1) == 0)  # [num_processes]，True表示该process的episode结束

            # 1. 批量构建 critic map input 和 critic vec input
            entropy_maps, _, _, all_critic_map_inp = stack_global_maps(
                self.envs_info, self.device
            )
            goal_done_mask = build_goal_done_mask(self.envs_info, self.device)
            agent_batterys = agent_batteries(
                self.envs_info, num_agents, self.device
            )

            # 3. 批量生成 landmark node
            detected_maps = build_detected_maps(self.envs_info, self.device)
            
            new_detected, new_detected_masks = self.update_landmark_info(all_landmark_datas, all_landmark_masks, detected_maps, self.device, env_dones) 
            # [num_agents * num_processes, max_landmarks, 4], [num_agents * num_processes, max_landmarks, 1]

            # 4. 批量通过RTT生成K个候选目标点
            ## 生成两张地图
            agent_entropy_maps = entropy_maps.unsqueeze(1).repeat(1, num_agents, 1, 1)  # [num_processes, num_agents, H, W]
            voronoi_masks_t = build_voronoi_masks(
                self.envs_info, num_agents, self.device
            )

            ## 从all_obs批量生成智能体信息（栅格索引） [num_processes, num_agents, 2]
            agent_context = build_agent_context(
                last_obs, last_goals, last_masks,
                agent_batterys, num_agents, num_processes
            )
            agent_positions = agent_context["agent_positions"]
            agent_nodes = agent_context["agent_nodes"]
            ego_nodes = agent_context["ego_nodes"]
            teammate_nodes = agent_context["teammate_nodes"]
            global_teammate_mask = agent_context["teammate_mask"]

            high_decision = self.high_level_policy.select_goals(
                policy=policy,
                obs_positions=last_obs[:, 2:4],
                all_masks=None,
                goals=last_goals,
                tasks=None,
                goal_log_probs=None,
                goal_done_mask=goal_done_mask,
                agent_entropy_maps=agent_entropy_maps,
                voronoi_masks=voronoi_masks_t,
                agent_nodes=agent_nodes,
                ego_nodes=ego_nodes,
                teammate_nodes=teammate_nodes,
                teammate_mask=global_teammate_mask,
                landmark_data=new_detected,
                landmark_mask=new_detected_masks,
                num_processes=num_processes,
                deterministic=False,
                update_tasks=False,
                update_log_probs=False,
                update_targeted=False,
                store_landmark_nodes=False,
            )
            last_goals = high_decision["goals"]
            
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
        self.rollout_writer.insert_transition(
            self.all_agents,
            obs,
            reward,
            high_rewards,
            masks,
            env_state,
            goal_dones,
        )

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, env_states, masks, goals, tasks, landmark_data, landmark_mask, deterministic=True):
        return self.eval_controller.act(
            self,
            obs,
            env_states,
            masks,
            goals,
            tasks,
            landmark_data,
            landmark_mask,
            deterministic=deterministic,
        )

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
