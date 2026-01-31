import torch
import torch.nn as nn
import torch.nn.functional as F
import copy  # 必须引入
import torch.optim as optim

class Hi_MADDPG:
    def __init__(self,
                 actor_critic,
                 # hyperparameters
                 tau,
                 gamma=0.99,
                 lr_a=1e-4,
                 lr_c=1e-3):
        
        self.tau = tau
        self.gamma = gamma
        self.lr_ac = lr_a
        self.lr_cc = lr_c
        # =================================================================
        # 1. High-Level Modules (Global Coordinator)
        # =================================================================

        self.main_ac = actor_critic
        self.target_ac = copy.deepcopy(actor_critic)

        self.high_net = self.main_ac.modules_dict['dth_high_level']
        self.target_high_net = self.target_ac.modules_dict['dth_high_level']

        self.optimizer = optim.Adam(self.high_net.parameters(), lr=self.lr_ac)

    # =================================================================
    # 2. Action Selection Methods
    # ================================================================= 

    def select_high_action(self, map_input, uav_pos):
        """
        高层决策：不涉及随机探索，通常直接输出
        Input: 
            map_input: [Batch, H, W]
            uav_pos: [Batch, N * 2]
        Output: 
            regions: [Batch, N, H, W]
            goals: [Batch, N, goal_dim]
        """
        with torch.no_grad():
            regions, goals, _ = self.main_ac.dth_high_level_act(uav_pos, map_input)
            
        return regions.cpu().numpy(), goals.cpu().numpy()

    def update(self, sample):
        """
        训练高层策略 (Actor-Critic)

        sample_data = {
            'global_state': self._to_tensor(self.global_state_buf[idxs]),
            'obs': self._to_tensor(self.obs_buf[idxs]),
            'R_u': self._to_tensor(self.R_u_buf[idxs]),
            'g_u': self._to_tensor(self.g_u_buf[idxs]),
            'rew': self._to_tensor(self.rew_buf[idxs]),
            'next_global_state': self._to_tensor(self.next_global_state_buf[idxs]),
            'next_obs': self._to_tensor(self.next_obs_buf[idxs]),
            'done': self._to_tensor(self.done_buf[idxs])
        }
        """
        # Unpack sample
        global_states = sample['global_state']  # [Batch, global_state_dim]
        obs = sample['obs']                    # [Batch, N, obs_dim]
        R_u = sample['R_u']                    # [Batch, N, region_dim]
        g_u = sample['g_u']                    # [Batch, N, goal_dim]
        rewards = sample['rew']                 # [Batch, N]
        next_global_states = sample['next_global_state']  # [Batch, global_state_dim]
        next_obs = sample['next_obs']          # [Batch, N, obs_dim]
        dones = sample['done']                 # [Batch, N]

        uav_pos = obs[:, :, 2:4].reshape(obs.shape[0], -1)  # [Batch, N * 2]
        next_uav_pos = next_obs[:, :, 2:4].reshape(obs.shape[0], -1)  # [Batch, N * 2]

        # --- 1. Critic Update (Value Loss) ---
        # compute target Q values
        with torch.no_grad():
            _, _, next_q = self.target_ac.dth_high_level_act(next_uav_pos, next_global_states)

            global_rewards = rewards.mean(dim=1, keepdim=True).view(-1, 1) / 1000  # [Batch, 1]
            global_dones = dones.all(dim=1, keepdim=True).float().view(-1, 1) # [Batch, 1]

            target_q = global_rewards + self.gamma * (1 - global_dones) * next_q
        
        # compute current Q values
        _, _, current_q = self.main_ac.dth_high_level_act(uav_pos, global_states)
        # Critic Loss
        critic_loss = F.mse_loss(current_q, target_q)

        # --- 2. Actor Update (Policy Loss) ---
        actor_loss = -current_q.mean()

        # --- 3. Optimize ---
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()

        # gradient clip
        torch.nn.utils.clip_grad_norm_(self.high_net.parameters(), 0.5)
        self.optimizer.step()

        # Soft Update Target
        self._soft_update(self.target_high_net, self.high_net, self.tau)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Lo_MADDPG:
    def __init__(self,
                 actor_critic,   # 传入包含 ModuleDict 的大模型
                 num_agents,
                 tau,
                 action_dim=5,
                 gamma=0.99,
                 target_update_freq=4,
                 lr_actor=1e-4,
                 lr_critic=1e-3):
        
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.train_step = 0
        self.target_update_freq = target_update_freq # 建议设为 2 或 3

        # =================================================================
        # 1. 提取底层模块 (Low-Level Modules Isolation)
        # =================================================================
        self.main_ac = actor_critic
        self.target_ac = copy.deepcopy(actor_critic)

        # 提取 Main 网络组件
        self.main_low = self.main_ac.modules_dict['dth_low_level']
        self.actor = self.main_low['dth_low_actor']
        self.critic_cnn = self.main_low['dth_low_map_cnn']
        self.critic_mlp = self.main_low['dth_low_vec_mlp']
        self.critic_heads = self.main_low['dth_low_value_heads']

        # 提取 Target 网络组件
        self.target_low = self.target_ac.modules_dict['dth_low_level']
        self.target_actor = self.target_low['dth_low_actor']
        self.target_critic_cnn = self.target_low['dth_low_map_cnn']
        self.target_critic_mlp = self.target_low['dth_low_vec_mlp']
        self.target_critic_heads = self.target_low['dth_low_value_heads']

        # =================================================================
        # 2. 定义优化器
        # =================================================================
        # 2.1 Actor 优化器 (只有1个，因为所有Agent共享同一个网络参数)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # 2.2 Critic 优化器 (包含 CNN, MLP 和所有 Heads)
        # 将所有 Critic 相关的参数收集起来
        critic_params = list(self.critic_cnn.parameters()) + \
                        list(self.critic_mlp.parameters()) + \
                        list(self.critic_heads.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)

    # =================================================================
    # 3. 辅助函数：手动拼接 Critic 前向传播
    # =================================================================
    def _critic_forward(self, is_target, map_input, uav_pos_n, goal_n, act_n):
        """
        手动执行: CNN(Map) + MLP(Vectors) -> Fusion -> Heads
        is_target: Bool, 是否使用 target 网络
        """
        # 选择网络组件
        if is_target:
            cnn, mlp, heads = self.target_critic_cnn, self.target_critic_mlp, self.target_critic_heads
        else:
            cnn, mlp, heads = self.critic_cnn, self.critic_mlp, self.critic_heads

        batch_size = map_input.size(0)

        # 1. 处理地图 (CNN)
        # map_input: [Batch, 1, H, W]
        map_feat = cnn(map_input).reshape(batch_size, -1) # Flatten

        # 2. 处理向量 (MLP)
        # 需要把所有 Agent 的 (obs, goal, act) 拼成一条长向量
        # uav_pos_n: [Batch, N, dim] -> [Batch, N*dim]
        uav_pos_flat = uav_pos_n.reshape(batch_size, -1)
        goal_flat = goal_n.reshape(batch_size, -1)
        act_flat = act_n.reshape(batch_size, -1)
        
        # 拼接顺序必须与 total_vec_dim 计算方式一致
        # 你的定义是: (4 + goal + act) * N
        # 这里最稳妥的方式是按 Agent 顺序交替拼接，或者全部 Flatten 后拼接
        # 假设 MLP 期望的是: [Agent1_all, Agent2_all, ...] 或 [All_obs, All_goals, All_acts]
        # 根据 total_vec_dim 的计算，这里我们将它们全部拼接
        vec_input = torch.cat([uav_pos_flat, goal_flat, act_flat], dim=-1)
        vec_feat = mlp(vec_input)

        # 3. 融合
        fused_feat = torch.cat([map_feat, vec_feat], dim=-1)

        # 4. 多头输出 Q 值
        q_values = []
        for i in range(self.num_agents):
            q_i = heads[i](fused_feat) # [Batch, 1]
            q_values.append(q_i)
        
        # [Batch, N]
        return torch.cat(q_values, dim=1)

    # =================================================================
    # 4. 动作选择 (Inference)
    # =================================================================
    def select_low_actions(self, uav_pos_n, goals_n, temperature=0.1, hard=True):
        
        actions_list = []
        with torch.no_grad():
            for i in range(self.num_agents):
                # 调用共享的 Actor
                # dth_low_level_act 是定义在 actor_critic 类里的方法，
                # 但我们需要用 self.actor (即 modules_dict['dth_low_actor']) 来计算
                # 注意：你的 dth_low_level_act 定义在类里，使用了 self.dth_low_actor
                # 这里我们直接调用 Main AC 的方法，或者模仿那个逻辑
                
                # 方案 A: 使用 Main AC 的方法 (最简单)
                # action_out, _ = self.main_ac.dth_low_level_act(obs_n[:, i], goals_n[:, i], temperature, hard)
                
                # 方案 B: 手动实现以解耦 (推荐，因为我们提取了 self.actor)
                input_tensor = torch.cat([uav_pos_n[:, i], goals_n[:, i]], dim=-1)
                logits = self.actor(input_tensor)
                
                if hard: # 测试/评估时通常用 Argmax One-hot
                    idx = torch.argmax(logits, dim=-1)
                    action_out = F.one_hot(idx, num_classes=self.action_dim).float()
                else: # 训练采样时
                    action_out = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
                
                actions_list.append(action_out)
        
        return torch.stack(actions_list, dim=1).cpu().numpy()

    # =================================================================
    # 5. 训练循环 (Update)
    # =================================================================
    def update(self, sample):
        """
        sample_data = {
            'obs': self._to_tensor(self.obs_buf[idxs]),            # [batch, num_agents, dim]
            'global_state': self._to_tensor(self.global_state_buf[idxs]),  # [batch, H, W]
            'goal': self._to_tensor(self.goal_buf[idxs]),          # [batch, num_agents, dim]
            'act': self._to_tensor(self.act_buf[idxs]),            # [batch, num_agents, dim]
            'rew': self._to_tensor(self.rew_buf[idxs]),            # [batch, num_agents, 1]
            'next_obs': self._to_tensor(self.next_obs_buf[idxs]),  # [batch, num_agents, dim]
            'next_global_state': self._to_tensor(self.next_global_state_buf[idxs]), # [batch, H, W]
            'done': self._to_tensor(self.done_buf[idxs])           # [batch, num_agents, 1]
        }
        """
        self.train_step += 1 # 每次调用计数 +1
        # 1. 解包数据
        obs_n = sample['obs']            # [B, N, obs_dim]
        curr_map = sample['global_state']# [B, H, W]
        goal_n = sample['goal']          # [B, N, goal_dim]
        act_n = sample['act']            # [B, N, action_dim]
        rewards_n = sample['rew']        # [B, N]
        next_obs_n = sample['next_obs']  # [B, N, obs_dim]
        next_map = sample['next_global_state'] # [B, H, W]
        dones_n = sample['done']         # [B, N]

        uav_state_n = obs_n[:, :, :4]          # [B, N, 2]
        next_uav_state_n = next_obs_n[:, :, :4]# [B, N, 2]
        next_goal_n = goal_n.clone()           # [B, N, goal_dim]

        # =============================================================
        # 1. Update Critic
        # =============================================================
        with torch.no_grad():
            # A. 计算 Target Actions (使用 Target Actor)
            next_acts_list = []
            for i in range(self.num_agents):
                # 构造输入
                inp = torch.cat([next_uav_state_n[:, i], next_goal_n[:, i]], dim=-1)
                logits = self.target_actor(inp)
                # 计算 Next Action (Hard Gumbel or Argmax)
                # 训练 Critic 时，Next Action 应尽可能接近真实策略
                next_act = F.gumbel_softmax(logits, tau=0.1, hard=True, dim=-1)
                next_acts_list.append(next_act)
            next_acts_tensor = torch.stack(next_acts_list, dim=1) # [B, N, Act_Dim]

            # B. 计算 Target Q (手动调用 Target Critic 组件)
            target_qs = self._critic_forward(True, next_map.unsqueeze(1), next_uav_state_n, next_goal_n, next_acts_tensor)
            
            # C. TD Target
            target_y = rewards_n + self.gamma * (1 - dones_n) * target_qs.unsqueeze(-1) # [B, N, 1]

        # D. Current Q (手动调用 Main Critic 组件), [B, N]
        current_qs = self._critic_forward(False, curr_map.unsqueeze(1), uav_state_n, goal_n, act_n)
        
        # E. Critic Loss
        critic_loss = F.mse_loss(current_qs.unsqueeze(-1), target_y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_low.parameters(), 0.5)
        self.critic_optimizer.step()

        # =============================================================
        # 2. Update Actor (Shared Parameter Update)
        # =============================================================
        # 由于所有 Agent 共享同一个 Actor 网络，我们累加所有 Agent 的 Loss
        # 然后做一次 Update
        
        total_actor_loss = 0
        
        for i in range(self.num_agents):
            # A. 重新生成当前 Agent 的动作 (带梯度)
            input_tensor = torch.cat([uav_state_n[:, i], goal_n[:, i]], dim=-1)
            logits = self.actor(input_tensor)
            # 使用 Gumbel-Softmax (hard=False) 以保持梯度平滑流向 logits
            curr_act_i = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
            
            # B. 构造联合动作
            # 只有 Agent i 的动作是新的且带梯度的，其他 Agent 使用 Buffer 中的旧动作
            joint_acts_list = []
            for j in range(self.num_agents):
                if i == j:
                    joint_acts_list.append(curr_act_i)
                else:
                    joint_acts_list.append(act_n[:, j].detach())
            joint_acts_tensor = torch.stack(joint_acts_list, dim=1)
            
            # C. 计算 Q 值 (Main Critic)
            # 我们只需要拿到属于 Agent i 的那个 Q Head 的输出
            # 为了节省计算，我们可以只跑 Critic 的 Backbone 一次，或者直接调用 _critic_forward
            qs = self._critic_forward(False, curr_map.unsqueeze(1), uav_state_n, goal_n, joint_acts_tensor)
            q_i = qs[:, i]
            
            # D. 累加 Loss (-Q)
            total_actor_loss += -q_i.mean()

        # E. 统一优化 Actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # =============================================================
        # 3. Soft Update
        # =============================================================
        # 遍历底层所有参数进行更新
        if self.train_step % self.target_update_freq == 0:
            self._soft_update(self.target_low, self.main_low, self.tau)

        return critic_loss.item(), total_actor_loss.item() / self.num_agents

    def _soft_update(self, target, source, tau):
        # 这里的 source/target 是 nn.ModuleDict，包含 actor 和 critic 所有组件
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)