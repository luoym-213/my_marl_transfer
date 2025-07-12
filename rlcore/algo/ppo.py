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

    def update(self, rollouts_list):
        # rollouts_list - list of rollouts of agents which share self.actor_critic policy
        advantages_list = []
        for rollout in rollouts_list:
            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            advantages_list.append(advantages)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                # raise ('sampler not implemented for recurrent policies')
                seq_length = 30
                data_generator = recurrent_feed_foward_generator(rollouts_list, advantages_list, self.num_mini_batch)
            else:
                data_generator = magent_feed_forward_generator(rollouts_list, advantages_list, self.num_mini_batch)
            
            # all_samples = [sample for sample in data_generator]

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,\
                masks_batch, old_action_log_probs_batch, adv_targ = sample
                # Reshape to do in a single forward pass for all steps
                # 在循环网络模式下，需要重新组织数据
                if self.actor_critic.is_recurrent:
                    # 重新组织为序列格式
                    # 从seq_length和obs_batch两个值获得samples_per_batch
                    samples_per_batch = obs_batch.shape[0] // seq_length
                    T, N = seq_length, samples_per_batch  # 这些值需要从generator中获取                    
                    
                    # 逐时间步处理
                    values_list = []
                    action_log_probs_list = []
                    dist_entropy_sum = 0 # 初始化熵的总和
                    current_states = recurrent_hidden_states_batch
                    
                    for t in range(T):
                        values_t, action_log_probs_t, dist_entropy_t, current_states = self.actor_critic.evaluate_actions(
                            obs_batch[t*N:(t+1)*N],
                            current_states,
                            masks_batch[t*N:(t+1)*N],
                            actions_batch[t*N:(t+1)*N]
                        )
                        # values_t:[N, 1], action_log_probs_t:[N, 1], dist_entropy_t:[N, 1]
                        values_list.append(values_t)
                        action_log_probs_list.append(action_log_probs_t)
                        dist_entropy_sum += dist_entropy_t
                    
                    # values, action_log_probs, dist_entropy:[N*T, 1]
                    values = torch.cat(values_list, dim=0)
                    action_log_probs = torch.cat(action_log_probs_list, dim=0)
                    dist_entropy = dist_entropy_sum / T
                else:
                    # 非循环网络的原有处理方式
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                    )

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
    """
    # 确保序列长度小于 episode 长度
    assert seq_length < 50, "序列长度必须小于 episode 长度 (50)"
    
    # 获取每个 rollout 的基本信息
    num_steps, num_processes = rollouts_list[0].rewards.size()[0:2]
    num_agents = len(rollouts_list)
    episode_length = 50  # 每个 episode 的长度
    num_episodes = num_steps // episode_length  # 每个 rollout 中的 episode 数量
    
    # 计算每个 mini-batch 的大小
    # 总样本数 = 代理数 × 进程数 × episode 数
    total_samples = num_agents * num_processes * num_episodes
    samples_per_batch = total_samples // num_mini_batch
    
    # 为每个样本生成随机的起始点
    for _ in range(num_mini_batch):
        obs_batch = []
        recurrent_hidden_states_batch = []
        actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []
        
        # 为每个 batch 采样
        for _ in range(samples_per_batch):
            # 随机选择一个代理
            agent_idx = torch.randint(0, num_agents, (1,)).item()
            rollout = rollouts_list[agent_idx]
            advantages = advantages_list[agent_idx]
            
            # 随机选择一个进程
            process_idx = torch.randint(0, num_processes, (1,)).item()
            
            # 随机选择一个 episode
            episode_idx = torch.randint(0, num_episodes, (1,)).item()
            
            # 随机选择起始点（确保在 episode 内）
            start_idx = torch.randint(0, episode_length - seq_length, (1,)).item()
            start_idx += episode_idx * episode_length  # 调整为全局索引
            
            # 收集序列数据
            obs_seq = rollout.obs[start_idx:start_idx+seq_length, process_idx]
            # 对于循环网络，我们只需要序列的初始隐藏状态
            recurrent_hidden_states_seq = rollout.recurrent_hidden_states[start_idx:start_idx+1, process_idx]
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
        
        # 将列表转换为张量
        obs_batch = torch.stack(obs_batch)
        recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch).view(samples_per_batch, -1)
        actions_batch = torch.stack(actions_batch)
        value_preds_batch = torch.stack(value_preds_batch)
        return_batch = torch.stack(return_batch)
        masks_batch = torch.stack(masks_batch)
        old_action_log_probs_batch = torch.stack(old_action_log_probs_batch)
        adv_targ = torch.stack(adv_targ)
        
        # 展平时序维度和批次维度【T*N, tensor】
        T, N = seq_length, samples_per_batch
        obs_batch = _flatten_helper(T, N, obs_batch.transpose(0, 1))
        actions_batch = _flatten_helper(T, N, actions_batch.transpose(0, 1))
        value_preds_batch = _flatten_helper(T, N, value_preds_batch.transpose(0, 1))
        return_batch = _flatten_helper(T, N, return_batch.transpose(0, 1))
        masks_batch = _flatten_helper(T, N, masks_batch.transpose(0, 1))
        old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch.transpose(0, 1))
        adv_targ = _flatten_helper(T, N, adv_targ.transpose(0, 1))
        
        yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
              masks_batch, old_action_log_probs_batch, adv_targ     