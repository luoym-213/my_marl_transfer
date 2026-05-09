import torch


class LowLevelPolicy:
    def _low_value(self, x):
        return self.value_head(x) # h_dim -> h_dim -> 1

    def _low_policy(self, x): # h_dim -> h_dim
        return self.policy_head(x)

    def data_processing_low_level(self, inp, goals, teammate_rel_pos, teammate_masks):
        # inp: [num_agents*batch_size, dim_o]
        # goals: [num_agents*batch_size, 2], assigned goals for agents
        # teammate_rel_pos: [num_agents*batch_size, A-1, 2]
        # teammate_masks: [num_agents*batch_size, A-1]

        batch_size = inp.size(0)

        # 提取速度 [batch_size, 2]
        velocities = inp[:, 0:2]

        # 提取自身位置 [batch_size, 2]
        self_pos = inp[:, 2:4]

        # 计算与目标的相对位置 [batch_size, 2]
        relative_goal_pos = goals - self_pos

        if teammate_rel_pos is None or teammate_masks is None:
            teammate_rel_pos = torch.zeros(
                batch_size,
                self.num_agents - 1,
                2,
                device=inp.device,
                dtype=inp.dtype,
            )
            teammate_masks = torch.zeros(
                batch_size,
                self.num_agents - 1,
                device=inp.device,
                dtype=inp.dtype,
            )

        teammate_rel_pos = teammate_rel_pos.to(device=inp.device, dtype=inp.dtype)
        teammate_masks = teammate_masks.to(device=inp.device, dtype=inp.dtype)
        relative_other_agents_pos = teammate_rel_pos.view(batch_size, -1)

        # 拼接新的观测向量
        # [batch_size, 2 + 2 + (A-1)*2 + (A-1)]
        new_inp = torch.cat([
            velocities,                    # 速度 (2)
            relative_goal_pos,             # 与目标的相对位置 (2)
            relative_other_agents_pos,     # 与通信内队友的相对位置 ((A-1)*2)
            teammate_masks                 # 通信内队友mask (A-1)
        ], dim=1)

        return new_inp

    def low_level_act(self, inp, goals, teammate_rel_pos=None, teammate_masks=None, deterministic=False):
        """
        inp: [num_agents*batch_size, dim_o]
        state: [num_agents*batch_size, dim_h]
        goals: [num_agents*batch_size, 2], assigned goals for agents
        mask: [batch_size, 1], mask for actions
        
        """
        # 处理观测和目标，得到新的输入
        new_inp = self.data_processing_low_level(
            inp, goals, teammate_rel_pos, teammate_masks
        )

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

    def evaluate_low_actions(self, inp, goals, action, teammate_rel_pos=None, teammate_masks=None):
        new_inp = self.data_processing_low_level(
            inp, goals, teammate_rel_pos, teammate_masks
        )
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        dist = self.dist(self._low_policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy
    
    def get_low_value(self, inp, goals, teammate_rel_pos=None, teammate_masks=None):
        new_inp = self.data_processing_low_level(
            inp, goals, teammate_rel_pos, teammate_masks
        )
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        return value
