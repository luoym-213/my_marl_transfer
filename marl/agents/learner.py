import torch
from marl.algos import IPPO, JointPPO
from marl.agents.factory import setup_master as setup_master_from_factory
from marl.agents.updates import update_policies
from marl.controllers.eval_controller import EvalController
from marl.controllers.high_level_context import (
    agent_alive_masks,
    agent_batteries,
    build_agent_context,
    goal_done_mask as build_goal_done_mask,
    local_detections as build_local_detections,
)
from marl.controllers.high_level_policy import HighLevelPolicy
from marl.controllers.landmark_memory import LandmarkMemory
from marl.controllers.local_maps import (
    LocalMapBank,
    build_comm_mask,
    build_low_level_comm_features,
    flatten_agent_major,
)
from marl.controllers.rollout_writer import RolloutWriter
from marl.timing import TimingProfiler


def setup_master(args, env=None, return_env=False):
    return setup_master_from_factory(
        args, Learner, env=env, return_env=return_env
    )


class Learner(object):
    def __init__(self, args, teams_list, policies_list, env):
        self.timer = getattr(args, "timing_timer", TimingProfiler())
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
                                       use_clipped_value_loss=args.clipped_value_loss,
                                       timing_timer=self.timer) for policy in self.policies_list]
        self.device = args.device
        self.env = env
        self.envs_info = None
        self.high_level_interval = args.high_level_interval
        self.comm_dist = args.comm_dist
        self.sensor_dist = args.sensor_dist
        self.local_voronoi_scope = args.local_voronoi_scope
        self.local_map_bank = LocalMapBank(
            args.num_processes,
            args.num_agents,
            self.device,
        )
        self.landmark_memory = LandmarkMemory(self.device)
        self.high_level_policy = HighLevelPolicy(
            args.top_k, args.rrt_max_iter, self.device, timing_timer=self.timer
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

    def _ensure_local_map_bank(self, num_processes, num_agents):
        if (
            self.local_map_bank.num_processes != num_processes
            or self.local_map_bank.num_agents != num_agents
        ):
            self.local_map_bank = LocalMapBank(
                num_processes,
                num_agents,
                self.device,
            )

    def _env_dones_from_masks(self, masks, num_agents, num_processes):
        episode_dones = masks.view(num_agents, num_processes).transpose(0, 1)
        return episode_dones.sum(dim=1) == 0

    def _world_steps_tensor(self):
        return torch.tensor(
            [info["world_steps"] for info in self.envs_info],
            dtype=torch.float32,
            device=self.device,
        )

    def _prepare_local_policy_inputs(
        self,
        all_obs,
        all_goals,
        all_masks,
        prev_landmark_data,
        prev_landmark_mask,
        prev_landmark_timestamp,
        num_agents,
        num_processes,
        env_dones,
    ):
        with self.timer.time("prepare_local_policy_inputs"):
            self._ensure_local_map_bank(num_processes, num_agents)
            if env_dones.any():
                self.local_map_bank.reset(env_dones)

            goal_done_mask = build_goal_done_mask(self.envs_info, self.device)
            batteries = agent_batteries(self.envs_info, num_agents, self.device)
            agent_context = build_agent_context(
                all_obs,
                all_goals,
                all_masks,
                batteries,
                num_agents,
                num_processes,
            )
            agent_positions = agent_context["agent_positions"]
            alive_mask = agent_alive_masks(self.envs_info, num_agents, self.device)
            comm_mask = build_comm_mask(
                agent_positions,
                self.comm_dist,
                alive_mask=alive_mask,
            )
            local_detection_list = build_local_detections(self.envs_info)
            step_ids = self._world_steps_tensor()

            with self.timer.time("local_map_update"):
                self.local_map_bank.update_from_local_observations(
                    agent_positions,
                    local_detection_list,
                    step_ids,
                    self.sensor_dist,
                )
            with self.timer.time("local_map_fuse"):
                self.local_map_bank.fuse_by_timestamp(comm_mask)

            # Actor map inputs are per-agent local maps: [P, A, 100, 100].
            with self.timer.time("entropy_map_compute"):
                agent_entropy_maps = self.local_map_bank.compute_entropy_maps()
            with self.timer.time("voronoi_mask_compute"):
                local_voronoi_masks = self.local_map_bank.get_local_voronoi_masks(
                    agent_positions,
                    comm_mask,
                    self.local_voronoi_scope,
                    alive_mask=alive_mask,
                ).float()

            with self.timer.time("landmark_memory_update"):
                new_detected, new_detected_masks, new_detected_timestamps = (
                    self.update_landmark_info(
                        prev_landmark_data,
                        prev_landmark_mask,
                        prev_landmark_timestamp,
                        local_detection_list,
                        comm_mask,
                        step_ids,
                        self.device,
                        env_dones,
                    )
                )

            # Critic no longer reads env global maps. It uses an aggregate of
            # learner-owned local maps; env global maps remain debug/render only.
            critic_entropy = agent_entropy_maps.mean(dim=1)
            critic_heatmap = self.local_map_bank.get_agents_heatmap(agent_positions)
            critic_belief = self.local_map_bank.belief_maps.mean(dim=1)
            critic_map_input = torch.stack(
                [critic_entropy, critic_heatmap, critic_belief],
                dim=1,
            )

            high_teammate_masks = comm_mask.float().unsqueeze(-1)
            batch_indices = torch.arange(num_agents, device=self.device)
            high_teammate_masks[:, batch_indices, batch_indices, 0] = 0.0
            # Receiver-specific high-level teammate input:
            # [P, A(receiver), A(sender), 5]. Communication-out teammate nodes,
            # including self, are zeroed before they reach the graph network.
            high_teammate_nodes = (
                agent_context["teammate_nodes"].unsqueeze(1)
                * high_teammate_masks
            )

            low_rel_pos, low_masks = build_low_level_comm_features(
                agent_positions,
                comm_mask,
            )

            return {
                "goal_done_mask": goal_done_mask,
                "agent_context": agent_context,
                "agent_entropy_maps": agent_entropy_maps,
                "local_voronoi_masks": local_voronoi_masks,
                "comm_mask": comm_mask,
                "critic_map_input": critic_map_input,
                "landmark_data": new_detected,
                "landmark_mask": new_detected_masks,
                "landmark_timestamp": new_detected_timestamps,
                "high_teammate_nodes": high_teammate_nodes,
                "high_teammate_masks": high_teammate_masks,
                "low_rel_pos": low_rel_pos,
                "low_masks": low_masks,
            }

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
            all_landmark_timestamps = torch.cat([agent.rollouts.landmark_timestamps[step-1] for agent in team])

            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # 从all_masks中提取env_done信息，全0表示当前process的episode结束
            env_dones = self._env_dones_from_masks(
                all_masks, num_agents, num_processes
            )

            prepared = self._prepare_local_policy_inputs(
                all_obs,
                all_goals,
                all_masks,
                all_landmark_datas,
                all_landmark_masks,
                all_landmark_timestamps,
                num_agents,
                num_processes,
                env_dones,
            )

            goal_done_mask = prepared["goal_done_mask"]
            agent_entropy_maps = prepared["agent_entropy_maps"]
            voronoi_masks_t = prepared["local_voronoi_masks"]
            all_critic_map_inp = prepared["critic_map_input"]
            new_detected = prepared["landmark_data"]
            new_detected_masks = prepared["landmark_mask"]
            new_detected_timestamps = prepared["landmark_timestamp"]
            agent_context = prepared["agent_context"]
            agent_positions = agent_context["agent_positions"]
            agent_nodes = agent_context["agent_nodes"]
            ego_nodes = agent_context["ego_nodes"]
            comm_mask = prepared["comm_mask"]
            high_teammate_nodes = prepared["high_teammate_nodes"]
            high_teammate_masks = prepared["high_teammate_masks"]

            # landmark node docker: Tensor shape [num_agents * num_processes, Max_L, 4]
            all_landmark_nodes = torch.zeros(num_processes * num_agents, new_detected.shape[1], 4, device=self.device)

            with self.timer.time("high_select_goals"):
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
                    teammate_nodes=high_teammate_nodes,
                    teammate_mask=high_teammate_masks,
                    landmark_data=new_detected,
                    landmark_mask=new_detected_masks,
                    goal_visibility_mask=comm_mask,
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
            all_landmark_timestamps = new_detected_timestamps
            all_landmark_nodes = high_decision["landmark_nodes"]

            # ⭐ 准备rollout数据
            # 转换 ego_nodes 为 agent-major 顺序，以匹配后续的 chunk 操作
            # [P, A, D] -> [A, P, D] -> [A*P, D]
            all_ego_nodes = flatten_agent_major(ego_nodes)
            K = self.top_k
            # Receiver-specific teammate nodes are already communication-masked.
            all_teammate_nodes = flatten_agent_major(high_teammate_nodes)  # [num_agents * num_processes, num_agents, 5]
            all_teammate_masks = flatten_agent_major(high_teammate_masks)  # [num_agents * num_processes, num_agents, 1]
            all_low_rel_pos = flatten_agent_major(prepared["low_rel_pos"])  # [num_agents * num_processes, A-1, 2]
            all_low_masks = flatten_agent_major(prepared["low_masks"])  # [num_agents * num_processes, A-1]
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
            with self.timer.time("high_value_forward"):
                all_high_value = policy.get_high_value(all_critic_map_inp, all_critic_nodes) # 计算所有process的高层value： [num_processes, num_agents]

            # ⭐ 底层策略
            with self.timer.time("low_policy_forward"):
                props = policy.low_level_act(
                    all_obs,
                    all_goals,
                    all_low_rel_pos,
                    all_low_masks,
                    deterministic=False,
                )

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
                landmark_timestamp=all_landmark_timestamps,
                landmark_nodes=all_landmark_nodes,
                low_teammate_rel_pos=all_low_rel_pos,
                low_teammate_masks=all_low_masks,
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
                             prev_landmark_timestamp, local_detection_list,
                             comm_mask, step_ids, device, env_dones=None,
                             match_threshold=0.05):
        return self.landmark_memory.update(
            prev_landmark_data,
            prev_landmark_mask,
            prev_landmark_timestamp,
            local_detection_list,
            comm_mask,
            step_ids,
            device=device,
            env_dones=env_dones,
            match_threshold=match_threshold,
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
            all_landmark_timestamps = torch.cat([agent.rollouts.landmark_timestamps[-1] for agent in team])

            # 向量化优化=================
            # 输入提取
            ## 批量处理所有环境
            num_processes = len(self.envs_info)
            num_agents = len(team)

            # 从all_masks中提取env_done信息，全0表示当前process的episode结束
            env_dones = self._env_dones_from_masks(
                last_masks, num_agents, num_processes
            )

            prepared = self._prepare_local_policy_inputs(
                last_obs,
                last_goals,
                last_masks,
                all_landmark_datas,
                all_landmark_masks,
                all_landmark_timestamps,
                num_agents,
                num_processes,
                env_dones,
            )

            goal_done_mask = prepared["goal_done_mask"]
            all_critic_map_inp = prepared["critic_map_input"]
            agent_entropy_maps = prepared["agent_entropy_maps"]
            voronoi_masks_t = prepared["local_voronoi_masks"]
            new_detected = prepared["landmark_data"]
            new_detected_masks = prepared["landmark_mask"]
            agent_context = prepared["agent_context"]
            agent_positions = agent_context["agent_positions"]
            agent_nodes = agent_context["agent_nodes"]
            ego_nodes = agent_context["ego_nodes"]
            comm_mask = prepared["comm_mask"]
            high_teammate_nodes = prepared["high_teammate_nodes"]
            high_teammate_masks = prepared["high_teammate_masks"]

            with self.timer.time("high_select_goals"):
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
                    teammate_nodes=high_teammate_nodes,
                    teammate_mask=high_teammate_masks,
                    landmark_data=new_detected,
                    landmark_mask=new_detected_masks,
                    goal_visibility_mask=comm_mask,
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
                with self.timer.time("high_value_forward"):
                    next_high_value = policy.get_high_value(all_critic_map_inp, all_critic_nodes) # 计算所有process的高层value： [num_processes, num_agents]
                with self.timer.time("low_policy_forward"):
                    next_low_value = policy.get_low_value(
                        last_obs,
                        last_goals,
                        flatten_agent_major(prepared["low_rel_pos"]),
                        flatten_agent_major(prepared["low_masks"]),
                    )

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

    def eval_act(self, obs, env_states, masks, goals, tasks, landmark_data,
                 landmark_mask, landmark_timestamp, deterministic=True):
        return self.eval_controller.act(
            self,
            obs,
            env_states,
            masks,
            goals,
            tasks,
            landmark_data,
            landmark_mask,
            landmark_timestamp,
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
