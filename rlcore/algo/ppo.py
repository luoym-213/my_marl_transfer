import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch,
                                             recurrent_hidden_states_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped=value_preds_batch+(values-value_preds_batch).clamp(-self.clip_param,self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)
                
                self.optimizer.zero_grad()                
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class JointPPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False,
                 parallel_batch_size=4):  # 新增参数：并行处理的batch数量

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.parallel_batch_size = parallel_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    def update(self, rollouts_list):
        # rollouts_list - list of rollouts of agents which share self.actor_critic policy
        advantages_list = []
        for rollout in rollouts_list:
            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            advantages_list.append(advantages) # [num_steps, num_processes, 1]

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                # raise ('sampler not implemented for recurrent policies')
                data_generator = recurrent_feed_foward_generator(rollouts_list, advantages_list, self.num_mini_batch, seq_length = 30)
                #data_generator = recurrent_from_0_feed_foward_generator(rollouts_list, advantages_list, self.num_mini_batch)
            else:
                data_generator = magent_feed_forward_generator(rollouts_list, advantages_list, self.num_mini_batch)
            
            # 收集所有样本
            all_samples = list(data_generator) # list of tuples = num_mini_batch
            
            # 按parallel_batch_size分组处理
            for i in range(0, len(all_samples), self.parallel_batch_size):
                batch_group = all_samples[i:i+self.parallel_batch_size]
                
                # 合并多个batch
                # reshape and concat along second dimension (episodes_per_batch * parallel_batch_size), (num_agent, batch, num_steps, dim) 
                combined_obs = torch.cat([sample[0] for sample in batch_group], dim=1)  # concat on dim=1
                combined_hidden_states = torch.cat([sample[1] for sample in batch_group], dim=1)
                combined_actions = torch.cat([sample[2] for sample in batch_group], dim=1)
                combined_value_preds = torch.cat([sample[3] for sample in batch_group], dim=1)
                combined_returns = torch.cat([sample[4] for sample in batch_group], dim=1)
                combined_masks = torch.cat([sample[5] for sample in batch_group], dim=1)
                combined_old_log_probs = torch.cat([sample[6] for sample in batch_group], dim=1)
                combined_advantages = torch.cat([sample[7] for sample in batch_group], dim=1)

                # 转换成形状(num_steps, num_agent * batch, dim)
                combined_obs = combined_obs.permute(2, 0, 1, 3).reshape(-1, combined_obs.size(0) * combined_obs.size(1), combined_obs.size(3))
                combined_hidden_states = combined_hidden_states.permute(2, 0, 1, 3).reshape(-1, combined_hidden_states.size(0) * combined_hidden_states.size(1), combined_hidden_states.size(3))
                combined_actions = combined_actions.permute(2, 0, 1, 3).reshape(-1, combined_actions.size(0) * combined_actions.size(1), combined_actions.size(3))
                combined_value_preds = combined_value_preds.permute(2, 0, 1, 3).reshape(-1, combined_value_preds.size(0) * combined_value_preds.size(1), combined_value_preds.size(3))
                combined_returns = combined_returns.permute(2, 0, 1, 3).reshape(-1, combined_returns.size(0) * combined_returns.size(1), combined_returns.size(3))
                combined_masks = combined_masks.permute(2, 0, 1, 3).reshape(-1, combined_masks.size(0) * combined_masks.size(1), combined_masks.size(3))
                combined_old_log_probs = combined_old_log_probs.permute(2, 0, 1, 3).reshape(-1, combined_old_log_probs.size(0) * combined_old_log_probs.size(1), combined_old_log_probs.size(3))
                combined_advantages = combined_advantages.permute(2, 0, 1, 3).reshape(-1, combined_advantages.size(0) * combined_advantages.size(1), combined_advantages.size(3))
                
                
                ## 检测代码开始
                # print("pre_0;", batch_group[0][0][0, 0, 0, :])
                # print("pre_1:", batch_group[1][0][0, 0, 0, :])
                # print("pre_2:", batch_group[2][0][0, 0, 0, :])
                # print("pre_3:", batch_group[3][0][0, 0, 0, :])

                # print("combined_obs_0:", combined_obs[0, 0, :])
                # print("combined_obs_1:", combined_obs[0, 6, :])
                # print("combined_obs_1:", combined_obs[0, 12, :])
                # print("combined_obs_1:", combined_obs[0, 18, :])
                # print("combined_obs.shape:", combined_obs.size())
                ## 检测代码结束

                # 一次性处理合并后的大batch
                if self.actor_critic.is_recurrent:

                    T, N = combined_obs.shape[0], combined_obs.shape[1]  # 50, numagent * episodes_per_batch * parallel_batch_size

                    # 初始化            
                    values_list = []
                    action_log_probs_list = []
                    dist_entropy_sum = 0 # 初始化熵的总和
                    # current_states = torch.zeros(N, combined_hidden_states.shape[2], device=combined_hidden_states.device) #[batchsize, hidden_size]
                    current_states = combined_hidden_states[0] # 取第一个时间步的隐藏状态作为初始状态
                    
                    # 逐时间步处理
                    for t in range(T):
                        #print("current_states: ", current_states[0:3])
                        values_t, action_log_probs_t, dist_entropy_t, current_states = self.actor_critic.evaluate_actions(
                            combined_obs[t],
                            current_states,
                            combined_masks[t],
                            combined_actions[t]
                        )
                        # values_t:[N, 1], action_log_probs_t:[N, 1], dist_entropy_t:[N, 1]

                        values_list.append(values_t)
                        action_log_probs_list.append(action_log_probs_t) 
                        dist_entropy_sum += dist_entropy_t

                    # values, action_log_probs, dist_entropy:[T, N, 1]
                    values = torch.stack(values_list, dim=0) # [T, N, -1]
                    action_log_probs = torch.stack(action_log_probs_list, dim=0) # [T, N, -1]
                    dist_entropy = dist_entropy_sum / T
                else:
                    # 非循环网络的原有处理方式
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                    )

                ratio = torch.exp(action_log_probs - combined_old_log_probs) # [T, N, 1]
                surr1 = ratio * combined_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * combined_advantages
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped= combined_value_preds+(values-combined_value_preds).clamp(-self.clip_param,self.clip_param)
                    value_losses = (values - combined_returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - combined_returns).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(combined_returns, values)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
            
            # 在处理完所有batch组后更新参数
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch        


def magent_feed_forward_generator(rollouts_list, advantages_list, num_mini_batch):
    num_steps, num_processes = rollouts_list[0].rewards.size()[0:2]
    batch_size = num_processes * num_steps
    mini_batch_size = int((batch_size/num_mini_batch)) # size of minibatch for each agent
    sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    for indices in sampler:
        obs_batch=torch.cat([rollout.obs[:-1].view(-1,*rollout.obs.size()[2:])[indices] for rollout in rollouts_list],0)
        recurrent_hidden_states_batch = torch.cat([rollout.recurrent_hidden_states[:-1].view(-1, 
                    rollout.recurrent_hidden_states.size(-1))[indices] for rollout in rollouts_list],0)
        actions_batch = torch.cat([rollout.actions.view(-1,
                    rollout.actions.size(-1))[indices] for rollout in rollouts_list],0)
        value_preds_batch=torch.cat([rollout.value_preds[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        return_batch = torch.cat([rollout.returns[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        masks_batch = torch.cat([rollout.masks[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        old_action_log_probs_batch=torch.cat([rollout.action_log_probs.view(-1,1)[indices] for rollout in rollouts_list],0)
        adv_targ = torch.cat([advantages.view(-1, 1)[indices] for advantages in advantages_list],0)

        yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,\
              masks_batch, old_action_log_probs_batch, adv_targ

def recurrent_feed_foward_generator(rollouts_list, advantages_list, num_mini_batch, seq_length=30):
    """
    循环网络样本生成器，负责高效地准备和提供循环神经网络所需的序列数据
    
    Args:
        rollouts_list: 代理的经验池列表，长度为 num_agents，每个元素是一个 RolloutStorage 对象
        advantages_list: 优势函数列表，与 rollouts_list 对应
        num_mini_batch: mini-batch 的数量
        seq_length: 每个序列的长度，默认为 30，必须小于 50
    
    Returns:
        生成器，每次迭代返回一个 mini-batch 的数据    

    - agent的索引需要一致，也就是每个agent的rollouts和advantages需要一一对应
    - 所有的索引值是二元组 (process_idx, episode_idx)
    - 一共 num_mini_batch 个batch
    - 每个batch包含 num_processes * num_episodes * episodes_length / num_mini_batch / seq_length 个序列
    - 比如，num_processes=32，num_episodes=3，episodes_length=50，num_mini_batch=16，seq_length=30， 则共10个序列
    - 随机选取序列起点: 在每个batch开始时，随机抽取process_idx和episode_idx，再在内随机选择起始点
    - 批次内有序 : 每个batch内的episode是连续的，但是保持，agent0,agent1,agent2,...的顺序
    - 最后输出的batch形状为 [num_agents, seq_per_batch, seq_length, features]
    """
    # 确保序列长度小于 episode 长度
    assert seq_length < 50, "序列长度必须小于 episode 长度 (50)"
    
    # 获取每个 rollout 的基本信息
    num_steps, num_processes = rollouts_list[0].rewards.size()[0:2]
    num_agents = len(rollouts_list)
    episode_length = 50  # 每个 episode 的长度
    num_episodes = num_steps // episode_length  # 每个 rollout 中的 episode 数量
    
    # 计算每个 mini-batch 的大小
    # 总样本数 = 进程数 × num_steps // seq_length
    total_seq = num_processes * num_episodes * episode_length // seq_length
    seq_per_batch = total_seq // num_mini_batch
    
    # 遍历所有的batch
    for _ in range(num_mini_batch):
        obs_batch = []
        recurrent_hidden_states_batch = []
        actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []

        # 一次性生成seq_per_batch个随机进程和episode索引以及起始点
        process_indices = torch.randint(0, num_processes, (seq_per_batch,))
        episode_indices = torch.randint(0, num_episodes, (seq_per_batch,))
        start_indices = torch.randint(0, episode_length - seq_length, (seq_per_batch,))
        # 计算全局索引
        start_indices = episode_indices * episode_length + start_indices
        # 合并成索引对列表
        indices_list = list(zip(process_indices.tolist(), start_indices.tolist()))
        
        # 为每个 batch 采样 seq_per_batch 个序列
                  
        for rollout, advantages in zip(rollouts_list, advantages_list):
            for process_idx, start_idx in indices_list:
                # 收集序列数据
                obs_seq = rollout.obs[start_idx:start_idx+seq_length, process_idx]
                # 对于循环网络，我们只需要序列的初始隐藏状态
                recurrent_hidden_states_seq = rollout.recurrent_hidden_states[start_idx:start_idx+seq_length, process_idx]
                actions_seq = rollout.actions[start_idx:start_idx+seq_length, process_idx]
                value_preds_seq = rollout.value_preds[start_idx:start_idx+seq_length, process_idx]
                return_seq = rollout.returns[start_idx:start_idx+seq_length, process_idx]
                masks_seq = rollout.masks[start_idx:start_idx+seq_length, process_idx]
                action_log_probs_seq = rollout.action_log_probs[start_idx:start_idx+seq_length, process_idx]
                adv_seq = advantages[start_idx:start_idx+seq_length, process_idx]
            
                # 添加到 batch
                obs_batch.append(obs_seq)
                recurrent_hidden_states_batch.append(recurrent_hidden_states_seq)
                actions_batch.append(actions_seq)
                value_preds_batch.append(value_preds_seq)
                return_batch.append(return_seq)
                masks_batch.append(masks_seq)
                old_action_log_probs_batch.append(action_log_probs_seq)
                adv_targ.append(adv_seq)
        
        # 转换为张量并重塑维度为 [num_agents, seq_per_batch, seq_length, features]
        obs_batch = torch.stack(obs_batch).view(num_agents, seq_per_batch, seq_length, -1)
        recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch).view(num_agents, seq_per_batch, seq_length, -1)
        actions_batch = torch.stack(actions_batch).view(num_agents, seq_per_batch, seq_length, -1)
        value_preds_batch = torch.stack(value_preds_batch).view(num_agents, seq_per_batch, seq_length, -1)
        return_batch = torch.stack(return_batch).view(num_agents, seq_per_batch, seq_length, -1)
        masks_batch = torch.stack(masks_batch).view(num_agents, seq_per_batch, seq_length, -1)
        old_action_log_probs_batch = torch.stack(old_action_log_probs_batch).view(num_agents, seq_per_batch, seq_length, -1)
        adv_targ = torch.stack(adv_targ).view(num_agents, seq_per_batch, seq_length, -1)
        
        yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
              masks_batch, old_action_log_probs_batch, adv_targ  

def recurrent_from_0_feed_foward_generator(rollouts_list, advantages_list, num_mini_batch):
    """
    最小化打乱，但保证基本的随机性
    - rollouts_list: num_agents
    - advantages_list: num_agents
    - agent的索引需要一致，也就是每个agent的rollouts和advantages需要一一对应
    - 所有的索引值是二元组 (process_idx, episode_idx)，一共 num_processes * num_episodes 个索引
    - 一次性打乱 : 在每个epoch开始时，将索引组打乱一次
    - 一共 num_mini_batch 个batch，每个batch包含 num_processes * num_episodes / num_mini_batch 个episode
    - 顺序分配 : 然后按顺序将打乱后的episode分配给各个batch
    - 批次内有序 : 每个batch内的episode是连续的，但是保持，agent0,agent1,agent2,...的顺序
    - 最后输出的batch形状为 [num_agents, episodes_per_batch, episode_length, features]
    """
    num_steps, num_processes = rollouts_list[0].rewards.size()[0:2] # 150, 32
    num_agents = len(rollouts_list)
    episode_length = 50
    num_episodes = num_steps // episode_length
    
    # 生成所有episode的索引
    all_episodes = [] # num_processes * num_episodes
    for process_idx in range(num_processes):
        for episode_idx in range(num_episodes):
            all_episodes.append((process_idx, episode_idx))
    
    # 只在epoch开始时打乱一次，然后顺序分配给各个batch，[3,7,1,9,2,8,4,6,0,5]
    shuffled_episodes = torch.randperm(len(all_episodes)) 
    
    # 计算每个batch的大小
    total_episodes = len(all_episodes) # num_processes * num_episodes
    episodes_per_batch = total_episodes // num_mini_batch # num_processes * num_episodes / num_mini_batch
    
    for batch_idx in range(num_mini_batch):
        start_idx = batch_idx * episodes_per_batch
        end_idx = start_idx + episodes_per_batch
        
        # 最后一个batch包含剩余的所有episode
        if batch_idx == num_mini_batch - 1:
            end_idx = total_episodes
        
        batch_episode_indices = shuffled_episodes[start_idx:end_idx]
        current_batch_size = end_idx - start_idx
        
        obs_batch = []
        recurrent_hidden_states_batch = []
        actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []
        
        # 按照打乱后的顺序收集数据
        for rollout, advantages in zip(rollouts_list, advantages_list):
            for idx in batch_episode_indices:
                process_idx, episode_idx = all_episodes[idx] 
            
                # 从episode起点开始
                start_step = episode_idx * episode_length
                end_step = start_step + episode_length
            
                # 收集序列数据
                obs_seq = rollout.obs[start_step:end_step, process_idx]
                recurrent_hidden_states_seq = rollout.recurrent_hidden_states[start_step:end_step, process_idx]
                actions_seq = rollout.actions[start_step:end_step, process_idx]
                value_preds_seq = rollout.value_preds[start_step:end_step, process_idx]
                return_seq = rollout.returns[start_step:end_step, process_idx]
                masks_seq = rollout.masks[start_step:end_step, process_idx]
                action_log_probs_seq = rollout.action_log_probs[start_step:end_step, process_idx]
                adv_seq = advantages[start_step:end_step, process_idx]
                
                # 添加到batch
                obs_batch.append(obs_seq)
                recurrent_hidden_states_batch.append(recurrent_hidden_states_seq)
                actions_batch.append(actions_seq)
                value_preds_batch.append(value_preds_seq)
                return_batch.append(return_seq)
                masks_batch.append(masks_seq)
                old_action_log_probs_batch.append(action_log_probs_seq)
                adv_targ.append(adv_seq)

        # 转换为张量并重塑维度为 [num_agents, episodes_per_batch, episode_length, features]
        obs_batch = torch.stack(obs_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        actions_batch = torch.stack(actions_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        value_preds_batch = torch.stack(value_preds_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        return_batch = torch.stack(return_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        masks_batch = torch.stack(masks_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        old_action_log_probs_batch = torch.stack(old_action_log_probs_batch).view(num_agents, episodes_per_batch, episode_length, -1)
        adv_targ = torch.stack(adv_targ).view(num_agents, episodes_per_batch, episode_length, -1)
       
        yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
              masks_batch, old_action_log_probs_batch, adv_targ

def recurrent_generator(rollouts_list, advantages_list, num_mini_batch, seq_len):
    """
    rolloutlist: buffer with fields [obs, actions, value_preds, returns, masks, action_log_probs, hidden_states]
    shape: [T+1, N, ...]
    advantages: [T, N]
    num_mini_batch: number of mini batches per update
    seq_len: length of sequence per sample (e.g. 16)
    output: [num_agents, batchsize, seq_len, dim]
    """
    T, N = rollouts_list[0].rewards.size()[0:2]   # T=128, N=num_processes
    batch_size = N * T # total number of samples = 128*32
    # 每个process有 floor(T/seq_len) 段序列
    num_sequences = batch_size // seq_len # 128*32/16 = 256
    assert num_sequences >= num_mini_batch, "mini batch size too large"

    # 生成所有序列的起点索引
    indices = torch.randperm(num_sequences) # 打乱顺序 [3,7,1,9,2,8,4,6,0,5...] 共256个

    mini_batch_size = num_sequences // num_mini_batch # 每个batch包含的序列数，共256/32=8个

    for start in range(0, num_sequences, mini_batch_size): # 32个batch
        sampled_indices = indices[start:start+mini_batch_size] # 每个batch的序列起点索引，共8个

        agent_obs_batch, agent_actions_batch, agent_value_preds_batch = [], [], []
        agent_return_batch, agent_masks_batch, agent_old_action_log_probs_batch = [], [], []
        agent_adv_batch, agent_hidden_states_batch = [], []

        for agent_id, (rollouts, advantages) in enumerate(zip(rollouts_list, advantages_list)):

            obs_batch, actions_batch, value_preds_batch = [], [], []
            return_batch, masks_batch, old_action_log_probs_batch = [], [], []
            adv_batch, hidden_states_batch = [], []

            for idx in sampled_indices:
                process_id = idx // (T // seq_len)
                start_step = (idx % (T // seq_len)) * seq_len
                end_step = start_step + seq_len

                obs_batch.append(rollouts.obs[start_step:end_step, process_id]) # [seq_len, obs_dim]
                actions_batch.append(rollouts.actions[start_step:end_step, process_id]) # [seq_len, action_dim]
                value_preds_batch.append(rollouts.value_preds[start_step:end_step, process_id])
                return_batch.append(rollouts.returns[start_step:end_step, process_id])
                masks_batch.append(rollouts.masks[start_step:end_step, process_id])
                old_action_log_probs_batch.append(rollouts.action_log_probs[start_step:end_step, process_id])
                adv_batch.append(advantages[start_step:end_step, process_id])

                # 初始 hidden state 取序列开头的 h0
                hidden_states_batch.append(rollouts.recurrent_hidden_states[start_step, process_id]) # [hidden_size]

            agent_obs_batch.append(torch.stack(obs_batch, dim=0)) # [batchsize, seq_len, obs_dim]
            agent_actions_batch.append(torch.stack(actions_batch, dim=0))
            agent_value_preds_batch.append(torch.stack(value_preds_batch, dim=0))
            agent_return_batch.append(torch.stack(return_batch, dim=0))
            agent_masks_batch.append(torch.stack(masks_batch, dim=0))
            agent_old_action_log_probs_batch.append(torch.stack(old_action_log_probs_batch, dim=0))
            agent_adv_batch.append(torch.stack(adv_batch, dim=0))
            agent_hidden_states_batch.append(torch.stack(hidden_states_batch, dim=0)) # [batchsize, hidden_size]

        # output: [num_agents, batchsize, seq_len, dim]
        yield (torch.stack(agent_obs_batch, dim=0),
               torch.stack(agent_hidden_states_batch, dim=0), # [agent, batchsize, hidden_size]
               torch.stack(agent_actions_batch, dim=0),
               torch.stack(agent_value_preds_batch, dim=0),
               torch.stack(agent_return_batch, dim=0),
               torch.stack(agent_masks_batch, dim=0),
               torch.stack(agent_old_action_log_probs_batch, dim=0),
               torch.stack(agent_adv_batch, dim=0))