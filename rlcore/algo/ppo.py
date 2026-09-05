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

class IPPO():
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

        self.trainable_params = [
            param
            for param in actor_critic.get_module_params('low_level')
            if param.requires_grad
        ]
        self.optimizer = optim.Adam(self.trainable_params, lr=lr)

    def update(self, rollouts_list):
        # rollouts_list - list of rollouts of different agents
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
                # data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
                raise NotImplementedError('sampler not implemented for recurrent policies')
            else:
                data_generator = feed_forward_generator(rollouts_list, advantages_list, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ, goal = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_low_actions(obs_batch,
                                             goal, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                # 需要根据masks_batch来计算loss， only consider the timesteps where masks_batch==1
                action_loss = -torch.min(surr1, surr2)
                action_loss = (action_loss * masks_batch).sum() / masks_batch.sum()

                if self.use_clipped_value_loss:
                    value_pred_clipped=value_preds_batch+(values-value_preds_batch).clamp(-self.clip_param,self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    # 根据 masks_batch 计算 value loss
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped)
                    value_loss = (value_loss * masks_batch).sum() / masks_batch.sum()
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)
                
                self.optimizer.zero_grad()                
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
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
                 parallel_batch_size=4,
                 dense_critic=True):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.parallel_batch_size = parallel_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.trainable_params = [
            param
            for module_name in ('high_level', 'high_critic')
            for param in actor_critic.get_module_params(module_name)
            if param.requires_grad
        ]
        self.optimizer = optim.Adam(self.trainable_params, lr=lr)

    def update(self, rollouts_list, dense_critic=True):
        """只在 SMDP 决策点上重算 actor/critic，并跨 epoch 复用数据集。"""
        advantages_list = [
            rollout.high_returns[:-1] - rollout.high_values[:-1]
            for rollout in rollouts_list
        ]
        decision_data = build_smdp_decision_dataset(
            rollouts_list, advantages_list
        )
        if decision_data is None:
            return 0.0, 0.0, 0.0

        loss_totals = torch.zeros(3, device=decision_data[0].device)
        num_updates = 0

        for _ in range(self.ppo_epoch):
            data_generator = smdp_feed_forward_generator(
                decision_data, self.num_mini_batch
            )
            for sample in data_generator:
                (
                    critic_maps_batch,
                    critic_nodes_batch,
                    goals_batch,
                    higoal_log_probs_batch,
                    high_value_preds_batch,
                    high_return_batch,
                    high_adv_targ,
                    agent_ids_batch,
                    ego_nodes_batch,
                    explore_nodes_batch,
                    landmark_masks_batch,
                    landmark_nodes_batch,
                    teammate_nodes_batch,
                    teammate_masks_batch
                ) = sample

                (
                    high_values,
                    new_higoal_log_probs,
                    higoal_dist_entropy
                ) = self.actor_critic.evaluate_high_actions(
                    critic_maps_batch,
                    critic_nodes_batch,
                    goals_batch,
                    ego_nodes_batch,
                    explore_nodes_batch,
                    landmark_masks_batch,
                    landmark_nodes_batch,
                    teammate_nodes_batch,
                    teammate_masks_batch,
                    agent_ids_batch
                )

                if self.use_clipped_value_loss:
                    value_pred_clipped = high_value_preds_batch + (
                        high_values - high_value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (
                        high_values - high_return_batch
                    ).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - high_return_batch
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    ).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(
                        high_return_batch, high_values
                    )

                decision_ratio = torch.exp(
                    new_higoal_log_probs - higoal_log_probs_batch
                )
                decision_surr1 = decision_ratio * high_adv_targ
                decision_surr2 = torch.clamp(
                    decision_ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param
                ) * high_adv_targ
                decision_loss = -torch.min(
                    decision_surr1, decision_surr2
                ).mean()
                decision_entropy = higoal_dist_entropy.mean()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + decision_loss
                    - decision_entropy * self.entropy_coef
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.trainable_params, self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    loss_totals += torch.stack([
                        value_loss,
                        decision_loss,
                        decision_entropy
                    ])
                num_updates += 1

        mean_losses = (loss_totals / num_updates).detach().cpu().tolist()
        return tuple(mean_losses)


def feed_forward_generator(rollouts_list, advantages_list, num_mini_batch):
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
        goal = torch.cat([rollout.goals.view(-1, rollout.goals.size(-1))[indices] for rollout in rollouts_list],0)

        yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,\
              masks_batch, old_action_log_probs_batch, adv_targ, goal

def build_smdp_decision_dataset(rollouts_list, advantages_list):
    """一次性提取所有 agent rollout 中的高层决策点。"""
    fields = [[] for _ in range(14)]

    for agent_id, (rollout, advantages) in enumerate(
            zip(rollouts_list, advantages_list)):
        decision_mask = rollout.goal_dones[:-1].reshape(-1) > 0.5
        tensors = (
            rollout.critic_maps.reshape(
                -1, *rollout.critic_maps.size()[2:]
            ),
            rollout.critic_nodes.reshape(
                -1, *rollout.critic_nodes.size()[2:]
            ),
            rollout.goals.reshape(-1, rollout.goals.size(-1)),
            rollout.higoal_log_probs.reshape(-1, 1),
            rollout.high_values[:-1].reshape(-1, 1),
            rollout.high_returns[:-1].reshape(-1, 1),
            advantages.reshape(-1, 1),
            rollout.ego_nodes.reshape(-1, rollout.ego_nodes.size(-1)),
            rollout.explore_nodes.reshape(
                -1, *rollout.explore_nodes.size()[2:]
            ),
            rollout.landmark_masks[:-1].reshape(
                -1, *rollout.landmark_masks.size()[2:]
            ),
            rollout.landmark_nodes.reshape(
                -1, *rollout.landmark_nodes.size()[2:]
            ),
            rollout.teammate_nodes.reshape(
                -1, *rollout.teammate_nodes.size()[2:]
            ),
            rollout.teammate_masks.reshape(
                -1, *rollout.teammate_masks.size()[2:]
            )
        )

        target_field_indices = tuple(range(7)) + tuple(range(8, 14))
        selected_tensors = [
            tensor[decision_mask] for tensor in tensors
        ]
        for field_index, tensor in zip(
                target_field_indices, selected_tensors):
            fields[field_index].append(tensor)

        fields[7].append(torch.full(
            (selected_tensors[0].size(0),),
            agent_id,
            dtype=torch.long,
            device=rollout.high_values.device
        ))

    decision_data = tuple(torch.cat(field, dim=0) for field in fields)
    if decision_data[0].size(0) == 0:
        return None

    advantages = decision_data[6]
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-5)
    decision_data = (
        decision_data[:6] + (advantages,) + decision_data[7:]
    )
    return decision_data


def smdp_feed_forward_generator(decision_data, num_mini_batch):
    """重新打乱已构造好的决策点数据集，不再拼接全量 rollout。"""
    batch_size = decision_data[0].size(0)
    num_batches = min(max(1, num_mini_batch), batch_size)
    permutation = torch.randperm(
        batch_size, device=decision_data[0].device
    )

    for indices in torch.chunk(permutation, num_batches):
        yield tuple(tensor[indices] for tensor in decision_data)


def recurrent_feed_foward_generator(rollouts_list, advantages_list, num_mini_batch, seq_length=30):
    """
    循环网络样本生成器，负责高效地准备和提供循环神经网络所需的序列数据
    
    Args:
        rollouts_list: 代理的经验池列表，长度为 num_agents，每个元素是一个 RolloutStorage 对象
        advantages_list: 优势函数列表，与 rollouts_list 对应
        num_mini_batch: mini-batch 的数量
        seq_length: 每个子序列的长度，默认设置为16
    
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
        agent_adv_batch, agent_hidden_states_batch, agent_env_states_batch, agent_tasks_batch = [], [], [],  []

        for agent_id, (rollouts, advantages) in enumerate(zip(rollouts_list, advantages_list)):

            obs_batch, actions_batch, value_preds_batch = [], [], []
            return_batch, masks_batch, old_action_log_probs_batch = [], [], []
            adv_batch, hidden_states_batch, env_states_batch, tasks_batch = [], [], [],  []

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
                env_states_batch.append(rollouts.env_states[start_step:end_step, process_id])
                tasks_batch.append(rollouts.tasks[start_step:end_step, process_id])

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
            agent_env_states_batch.append(torch.stack(env_states_batch, dim=0))
            agent_tasks_batch.append(torch.stack(tasks_batch, dim=0))

        # output: [num_agents, batchsize, seq_len, dim]
        yield (torch.stack(agent_obs_batch, dim=0),
               torch.stack(agent_hidden_states_batch, dim=0), # [agent, batchsize, hidden_size]
               torch.stack(agent_actions_batch, dim=0),
               torch.stack(agent_value_preds_batch, dim=0),
               torch.stack(agent_return_batch, dim=0),
               torch.stack(agent_masks_batch, dim=0),
               torch.stack(agent_old_action_log_probs_batch, dim=0),
               torch.stack(agent_adv_batch, dim=0),
               torch.stack(agent_env_states_batch, dim=0),
               torch.stack(agent_tasks_batch, dim=0))