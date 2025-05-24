from rlcore.algo import PPO
from rlcore.storage import RolloutStorage
import torch
import torch.nn as nn


class Neo(object):

  def __init__(self, args, policy, obs_shape, action_space):
    super().__init__()

    self.obs_shape = obs_shape
    self.action_space = action_space
    self.actor_critic = policy
    self.rollouts = RolloutStorage(args.num_steps, args.num_processes, self.obs_shape, self.action_space, 
                                   recurrent_hidden_state_size=1)
    self.args = args
    self.trainer = PPO(self.actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr,max_grad_norm=args.max_grad_norm)

  def load_model(self, policy_state):
      self.actor_critic.load_state_dict(policy_state)

  def initialize_obs(self, obs):
    # this function is called at the start of episode
    self.rollouts.obs[0].copy_(obs)

  def update_rollout(self, obs, reward, mask):
    self.rollouts.insert(obs, self.states, self.action, self.action_log_prob, self.value, reward, mask)

  def act(self, step, deterministic=False):
    self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(self.rollouts.obs[step],
              self.rollouts.recurrent_hidden_states[step],self.rollouts.masks[step],deterministic=deterministic)
    return self.action

  def wrap_horizon(self, next_value):
    self.rollouts.compute_returns(next_value, True, self.args.gamma, self.args.tau)

  def after_update(self):
    self.rollouts.after_update()

  def update(self):
    return self.trainer.update(self.rollouts)
  
  def filter_obs(self, step, obs, input_size = 4):
    # obs:[32,14]
    # 仅提取entity的obs
    # 新增实体过滤逻辑
    num_entities = self.args.num_agents
    mask_obs_dist = self.args.mask_obs_dist

    entity_inp = obs[:, input_size:input_size+self.args.num_agents*2]
  
    # 将entity_inp重塑为[processes, num_entities, 2]
    entities = entity_inp.view(-1, num_entities, 2)
    
    # 计算相对距离（L2范数）
    distances = torch.norm(entities, p=2, dim=2)  # [processes, num_entities]
    
    # 生成观测掩码
    valid_mask = (distances <= mask_obs_dist)
    # 构建带标签的观测列表
    labeled_entities = torch.cat([
        entities,
        torch.zeros_like(entities[:, :, :1])  # 添加label维度，初始化为0
    ], dim=2)  # [processes, num_entities, 3]
    
    # 应用掩码过滤
    valid_observations = [
        [entity.tolist() for entity, valid in zip(process_entities, process_mask) if valid]
        for process_entities, process_mask in zip(labeled_entities, valid_mask)
    ]
    
    # 收集有效观测数据
    for process_idx, observations in enumerate(valid_observations):
        self.rollouts.obs_valid.collect_observations(step, process_idx, observations)
    
    # return valid_observations
    