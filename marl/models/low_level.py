import torch


class LowLevelPolicy:
    def _low_value(self, x):
        return self.value_head(x) # h_dim -> h_dim -> 1

    def _low_policy(self, x): # h_dim -> h_dim
        return self.policy_head(x)

    def data_processing_low_level(self, inp, goals):
        # inp: [num_agents*batch_size, dim_o]
        # goals: [num_agents*batch_size, 2], assigned goals for agents

        batch_size = inp.size(0)

        # 提取速度 [batch_size, 2]
        velocities = inp[:, 0:2]

        # 提取自身位置 [batch_size, 2]
        self_pos = inp[:, 2:4]

        # 计算与目标的相对位置 [batch_size, 2]
        relative_goal_pos = goals - self_pos

        # 提取其他智能体的绝对位置
        # 从 inp 中提取：跳过速度(2)、自身位置(2)、landmarks(num_agents*2)
        other_agents_start_idx = 4 + self.num_agents * 2
        other_agents_pos = inp[:, other_agents_start_idx:other_agents_start_idx + (self.num_agents - 1) * 2]

        # 将其他智能体位置重塑为 [batch_size, num_agents-1, 2]
        other_agents_pos = other_agents_pos.view(batch_size, self.num_agents - 1, 2)

        # 计算与其他智能体的相对位置
        # 扩展 self_pos 以便广播: [batch_size, 1, 2]
        self_pos_expanded = self_pos.unsqueeze(1)

        # 相对位置 [batch_size, num_agents-1, 2]
        relative_other_agents_pos = other_agents_pos - self_pos_expanded

        # 展平其他智能体的相对位置 [batch_size, (num_agents-1)*2]
        relative_other_agents_pos = relative_other_agents_pos.view(batch_size, -1)

        # 拼接新的观测向量
        # [batch_size, 2 + 2 + (num_agents-1)*2]
        new_inp = torch.cat([
            velocities,                    # 速度 (2)
            relative_goal_pos,             # 与目标的相对位置 (2)
            relative_other_agents_pos      # 与其他智能体的相对位置 ((num_agents-1)*2)
        ], dim=1)

        return new_inp

    def low_level_act(self, inp, goals, deterministic=False):
        """
        inp: [num_agents*batch_size, dim_o]
        state: [num_agents*batch_size, dim_h]
        goals: [num_agents*batch_size, 2], assigned goals for agents
        mask: [batch_size, 1], mask for actions
        
        """
        # 处理观测和目标，得到新的输入
        new_inp = self.data_processing_low_level(inp, goals)  

        # 前向传播
        x = self.low_agent_encoder(new_inp)  # should be [batch_size, h_dim]
        value = self._low_value(x)  # should be [batch_size, 1]

        # 采样动作
        dist = self.dist(self._low_policy(x))
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1,1)

        return value, action, action_log_probs

    def evaluate_low_actions(self, inp, goals, action):
        new_inp = self.data_processing_low_level(inp, goals)
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        dist = self.dist(self._low_policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy
    
    def get_low_value(self, inp, goals):
        new_inp = self.data_processing_low_level(inp, goals)
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        return value
