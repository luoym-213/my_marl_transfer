import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, num_agent, recurrent_hidden_state_size):
        # 环境基础信息
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.env_states = torch.zeros(num_steps + 1, num_processes, num_agent*6)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

        # 低层策略存储相关
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1)
        self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        
        # 高层策略存储相关
        self.high_rewards = torch.zeros(num_steps, num_processes, 1)
        self.map_obs = torch.zeros(num_steps, num_processes, 4, 100, 100)  # 高层输入的地图观测
        self.vec_obs = torch.zeros(num_steps, num_processes, 5)  # 高层输入的矢量观测
        self.critic_maps = torch.zeros(num_steps, num_processes, 4, 100, 100)  # 高层critic的地图输入,用于计算高层价值
        self.tasks = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)   # action_mode: explore=0, collect=1
        self.goals = torch.zeros(num_steps, num_processes, 2)   # final_target: [x_goal, y_goal]
        self.map_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.decision_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.high_values = torch.zeros(num_steps + 1, num_processes, 1)
        self.high_returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.goal_dones = torch.zeros(num_steps + 1, num_processes, 1) # 当前是否是决策点，yes=1, no=0
        self.goal_dones[0].fill_(1.0) # 初始化第一步为1，因为需要执行高层策略来分配初始目标

    def to(self, device):
        # 环境基础信息
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.env_states = self.env_states.to(device)
        self.masks = self.masks.to(device)

        # 低层策略存储相关
        self.rewards = self.rewards.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        
        # 高层策略存储相关
        self.high_rewards = self.high_rewards.to(device)
        self.map_obs = self.map_obs.to(device)
        self.vec_obs = self.vec_obs.to(device)
        self.critic_maps = self.critic_maps.to(device)
        self.goals = self.goals.to(device)
        self.tasks = self.tasks.to(device)
        self.map_log_probs = self.map_log_probs.to(device)
        self.decision_log_probs = self.decision_log_probs.to(device)
        self.high_values = self.high_values.to(device)
        self.high_returns = self.high_returns.to(device)
        self.goal_dones = self.goal_dones.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, 
               rewards, high_rewards, masks, env_states, 
               map_obs, vec_obs, critic_maps,
               goals, task, 
               map_log_probs, decision_log_probs, 
               high_values, goal_dones):
        # 环境基础信息
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.env_states[self.step + 1].copy_(env_states)
        self.masks[self.step + 1].copy_(masks)

        # 低层策略存储相关
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        
        # 高层策略存储相关
        self.high_rewards[self.step].copy_(high_rewards)
        self.map_obs[self.step].copy_(map_obs)
        self.vec_obs[self.step].copy_(vec_obs)
        self.critic_maps[self.step].copy_(critic_maps)
        self.goals[self.step].copy_(goals)
        self.tasks[self.step].copy_(task)
        self.map_log_probs[self.step].copy_(map_log_probs)
        self.decision_log_probs[self.step].copy_(decision_log_probs)
        self.high_values[self.step].copy_(high_values)
        self.goal_dones[self.step + 1].copy_(goal_dones)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.env_states[0].copy_(self.env_states[-1])
        self.goal_dones[0].copy_(self.goal_dones[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma*self.value_preds[step+1]*self.masks[step+1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step+1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
    
    def compute_high_returns(self, next_high_value, gamma, tau):
        """
        计算高层策略的GAE回报，需要找到每个时间步的折扣更新
        goal_dones: 指示当前时间步是否为决策点，yes=1, no=0

        对于SMDP，需要累积决策点之间的折扣奖励，并在决策点处更新GAE
        """
        self.high_values[-1] = next_high_value
        
        # 初始化：这些变量维护"未来"的信息
        last_gae = torch.zeros_like(self.high_rewards[0])
        next_decision_value = next_high_value  # Buffer外的值V(s_T+1)
        accumulated_reward = torch.zeros_like(self.high_rewards[0])
        step_count = torch.zeros_like(self.high_rewards[0])  # 用于计算gamma^k，两个决策点之间的步数

        for step in reversed(range(self.high_rewards.size(0))):
            # 1. 处理环境Done（mask=0表示episode结束）
            if step < self.high_rewards.size(0) - 1: # 非最后一步
                # 如果环境结束，重置所有累积量,如果 step+1 是 done，那么未来的链条断开
                accumulated_reward = accumulated_reward * self.masks[step + 1]
                step_count = step_count * self.masks[step + 1]
                next_decision_value = next_decision_value * self.masks[step + 1]
                last_gae = last_gae * self.masks[step + 1]
        
            # 2. 累积奖励（所有步都累积，包括决策点和非决策点）
            accumulated_reward = self.high_rewards[step] + gamma * accumulated_reward
            step_count = step_count + 1
            
            # 3. 计算低层critic的returns（dense，每步都更新）
            # 这部分保持原有的逻辑，用于低层策略
            # self.high_returns[step] = ... (如果需要的话)
            
            # 4. 计算高层actor的advantage（sparse，仅在决策点）
            if self.goal_dones[step] > 0.5:  # 这是一个决策点
                curr_value = self.high_values[step]
                
                # 折扣因子：gamma^k，k是从当前决策点到下一个决策点的步数
                discount_factor = gamma ** step_count  # step_count已经+1了，所以-1
                # GAE衰减因子: gamma^k * tau
                gae_discount = gamma ** step_count * tau
                
                # TD误差：累积奖励 + 折扣的下一个决策点价值 - 当前价值
                # delta = R_acc + gamma^k * V_next - V_curr
                delta = accumulated_reward + discount_factor * next_decision_value * self.masks[step + 1] - curr_value
                
                # GAE更新
                # A_t = delta + (gamma * tau)^k * A_next
                gae = delta + gae_discount * last_gae

                # 存储返回值（高层策略的advantage）
                self.high_returns[step] = gae + curr_value
                
                # 重置，为上一个决策点做准备
                last_gae = gae
                next_decision_value = curr_value
                accumulated_reward = torch.zeros_like(self.high_rewards[0])
                step_count = torch.zeros_like(self.high_rewards[0])
            else:
                # 非决策点：advantage为0（或者不更新high_returns，保持为0）
                self.high_returns[step] = torch.zeros_like(self.high_values[step])
                # 注意：累积的reward和step_count会继续传递到上一个决策点


    def feed_forward_generator(self, advantages, num_mini_batch, sampler=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        if sampler is None:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, 
                                            self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ
