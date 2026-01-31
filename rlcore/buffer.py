import numpy as np
import torch

class LowLevelBuffer:
    def __init__(self, capacity, num_agents, obs_dim, act_dim, goal_dim, map_height=100, map_width=100, device='cpu'):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.device = device
        self.map_height = map_height
        self.map_width = map_width
        self.ptr = 0
        self.size = 0

        # Shape: [Capacity, num_agents, Obs_Dim]
        self.obs_buf = np.zeros((self.capacity, num_agents, obs_dim), dtype=np.float32)
        self.goal_buf = np.zeros((self.capacity, num_agents, goal_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity, num_agents, act_dim), dtype=np.float32)
        self.global_state_buf = np.zeros((self.capacity, map_height, map_width), dtype=np.uint8)
        
        # Reward 通常是 [Capacity, num_agents, 1] (每个智能体有自己的奖励)
        self.rew_buf = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)
        
        self.next_obs_buf = np.zeros((self.capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_global_state_buf = np.zeros((self.capacity, map_height, map_width), dtype=np.uint8)
        
        # Done 通常是 [Capacity, num_agents, 1] 或者是 [Capacity, 1] (取决于环境是各自结束还是一起结束)
        # MPE通常是一起结束，但为了兼容性，保留 N 维度
        self.done_buf = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)

    def add_batch(self, obs, global_state, goal, act, rew, next_obs, next_global_state, done):
        """
        obs shape: (batch_size, num_agents, obs_dim) -> from num_processes sampling
        goal shape: (batch_size, num_agents, goal_dim)
        act shape: (batch_size, num_agents, act_dim)
        rew shape: (batch_size, num_agents, 1)
        next_obs shape: (batch_size, num_agents, obs_dim)
        done shape: (batch_size, num_agents, 1)
        说明：
        1. 这里假设传入的 rew 和 done 是二维的 (batch_size, num_agents)，如果是一维的，需要在调用前扩展维度。
        2. 该函数会处理循环覆盖的情况。
        3. batch_size = num_processes
        """
        batch_size = len(obs)
        idx_end = self.ptr + batch_size
        
        if idx_end <= self.capacity:
            self.obs_buf[self.ptr : idx_end] = obs
            self.global_state_buf[self.ptr : idx_end] = global_state
            self.goal_buf[self.ptr : idx_end] = goal
            self.act_buf[self.ptr : idx_end] = act
            self.rew_buf[self.ptr : idx_end] = rew
            self.next_obs_buf[self.ptr : idx_end] = next_obs
            self.next_global_state_buf[self.ptr : idx_end] = next_global_state
            self.done_buf[self.ptr : idx_end] = done
        else:
            # 处理循环覆盖的逻辑：先填充缓冲区末尾，再从头部开始填充
            overflow = idx_end - self.capacity
            remain = self.capacity - self.ptr
            
            # 先填末尾 [ptr, capacity)
            self.obs_buf[self.ptr : self.capacity] = obs[:remain]
            self.global_state_buf[self.ptr : self.capacity] = global_state[:remain]
            self.goal_buf[self.ptr : self.capacity] = goal[:remain]
            self.act_buf[self.ptr : self.capacity] = act[:remain]
            self.rew_buf[self.ptr : self.capacity] = rew[:remain]
            self.next_obs_buf[self.ptr : self.capacity] = next_obs[:remain]
            self.next_global_state_buf[self.ptr : self.capacity] = next_global_state[:remain]
            self.done_buf[self.ptr : self.capacity] = done[:remain]
            
            # 再填开头 [0, overflow)
            self.obs_buf[0 : overflow] = obs[remain:]
            self.global_state_buf[0 : overflow] = global_state[remain:]
            self.goal_buf[0 : overflow] = goal[remain:]
            self.act_buf[0 : overflow] = act[remain:]
            self.rew_buf[0 : overflow] = rew[remain:]
            self.next_obs_buf[0 : overflow] = next_obs[remain:]
            self.next_global_state_buf[0 : overflow] = next_global_state[remain:]
            self.done_buf[0 : overflow] = done[remain:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, agent_id=None):
        """
        采样函数。
        如果 agent_id 为 None，返回所有智能体的数据 (用于 Critic)。
        如果指定 agent_id，返回特定智能体的数据 (用于 Actor 更新)。
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        data = {
            'obs': self._to_tensor(self.obs_buf[idxs]),            # [batch, num_agents, dim]
            'global_state': self._to_tensor(self.global_state_buf[idxs]),  # [batch, H, W]
            'goal': self._to_tensor(self.goal_buf[idxs]),          # [batch, num_agents, dim]
            'act': self._to_tensor(self.act_buf[idxs]),            # [batch, num_agents, dim]
            'rew': self._to_tensor(self.rew_buf[idxs]),            # [batch, num_agents, 1]
            'next_obs': self._to_tensor(self.next_obs_buf[idxs]),  # [batch, num_agents, dim]
            'next_global_state': self._to_tensor(self.next_global_state_buf[idxs]), # [batch, H, W]
            'done': self._to_tensor(self.done_buf[idxs])           # [batch, num_agents, 1]
        }
        
        # 如果你需要取特定 Agent 的数据来训练它的 Actor
        if agent_id is not None:
            return {k: v[:, agent_id] for k, v in data.items()}
            
        return data

    def _to_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)
    
class HighLevelBuffer:
    def __init__(self, capacity, num_agents, obs_dim, g_dim, map_height=100, map_width=100, device='cpu'):
        """
        高层Buffer，用于存储高层策略的经验数据
        
        参数:
            capacity: 缓冲区容量
            num_agents: 智能体数量
            global_state_dim: 全局状态维度
            map_height: 地图高度（栅格数）
            map_width: 地图宽度（栅格数）
            device: 设备（cpu或cuda）
        """
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.map_height = map_height
        self.map_width = map_width
        self.device = device
        self.ptr = 0
        self.size = 0

        # 1. state components
        # Shape: [Capacity, global_state_dim]: belief map 
        self.global_state_buf = np.zeros((self.capacity, map_height, map_width), dtype=np.uint8)
        
        # Shape: [Capacity, num_agents, obs_dim]（每个智能体的局部观察）
        self.obs_buf = np.zeros((self.capacity, num_agents, obs_dim), dtype=np.float32)
        
        # 2. action components
        # Shape: [Capacity, num_agents, H, W]（可达区域 R_u）
        self.R_u_buf = np.zeros((self.capacity, num_agents, map_height, map_width), dtype=np.uint8)

        # Shape: [Capacity, num_agents, g_dim]（每个智能体的任务向量）动作
        self.g_u_buf = np.zeros((self.capacity, num_agents, g_dim), dtype=np.float32)
        
        # 3. reward components
        # Shape: [Capacity, num_agents]（每个智能体的奖励）
        self.rew_buf = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)
        
        # 4. next state components
        # 下一状态
        self.next_global_state_buf = np.zeros((self.capacity, map_height, map_width), dtype=np.uint8)
        self.next_obs_buf = np.zeros((self.capacity, num_agents, obs_dim), dtype=np.float32)
        
        # 5. done components
        # Shape: [Capacity, num_agents, 1]（是否结束）
        self.done_buf = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)

    def add_batch(self, global_state, obs, R_u, g_u, rew, 
                  next_global_state, next_obs, done):
        """
        批量添加经验数据
        
        参数:
            global_state: 全局状态，形状 (batch_size, map_height, map_width)
            obs: 局部观察，形状 (batch_size, num_agents, obs_dim)
            R_u: 可达区域，形状 (batch_size, num_agents, map_height, map_width)
            g_u: 任务向量，形状 (batch_size, num_agents, g_dim)
            rew: 奖励，形状 (batch_size, num_agents, 1)
            next_global_state: 下一全局状态，形状 (batch_size, map_height, map_width)
            next_obs: 下一局部观察，形状 (batch_size, num_agents, obs_dim)
            done: 结束标志，形状 (batch_size, num_agents, 1)
         说明:
            1. 这里假设传入的 rew 和 done 是二维的 (batch_size, 1)，如果是一维的，需要在调用前扩展维度。
            2. 该函数会处理循环覆盖的情况。 
            3. batch_size = num_processes
        """
        batch_size = len(global_state)
        idx_end = self.ptr + batch_size
        
        if idx_end <= self.capacity:
            self.global_state_buf[self.ptr : idx_end] = global_state
            self.obs_buf[self.ptr : idx_end] = obs
            self.R_u_buf[self.ptr : idx_end] = R_u
            self.g_u_buf[self.ptr : idx_end] = g_u
            self.rew_buf[self.ptr : idx_end] = rew
            self.next_global_state_buf[self.ptr : idx_end] = next_global_state
            self.next_obs_buf[self.ptr : idx_end] = next_obs
            self.done_buf[self.ptr : idx_end] = done
        else:
            # 处理循环覆盖的逻辑
            overflow = idx_end - self.capacity
            remain = self.capacity - self.ptr
            
            # 先填末尾 [ptr, capacity)
            self.global_state_buf[self.ptr : self.capacity] = global_state[:remain]
            self.obs_buf[self.ptr : self.capacity] = obs[:remain]
            self.R_u_buf[self.ptr : self.capacity] = R_u[:remain]
            self.g_u_buf[self.ptr : self.capacity] = g_u[:remain]
            self.rew_buf[self.ptr : self.capacity] = rew[:remain]
            self.next_global_state_buf[self.ptr : self.capacity] = next_global_state[:remain]
            self.next_obs_buf[self.ptr : self.capacity] = next_obs[:remain]
            self.done_buf[self.ptr : self.capacity] = done[:remain]
            
            # 再填开头 [0, overflow)
            self.global_state_buf[0 : overflow] = global_state[remain:]
            self.obs_buf[0 : overflow] = obs[remain:]
            self.R_u_buf[0 : overflow] = R_u[remain:]
            self.g_u_buf[0 : overflow] = g_u[remain:]
            self.rew_buf[0 : overflow] = rew[remain:]
            self.next_global_state_buf[0 : overflow] = next_global_state[remain:]
            self.next_obs_buf[0 : overflow] = next_obs[remain:]
            self.done_buf[0 : overflow] = done[remain:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """
        随机采样一批数据
        
        返回:
            包含所有状态和奖励的字典
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        data = {
            'global_state': self._to_tensor(self.global_state_buf[idxs]),
            'obs': self._to_tensor(self.obs_buf[idxs]),
            'R_u': self._to_tensor(self.R_u_buf[idxs]),
            'g_u': self._to_tensor(self.g_u_buf[idxs]),
            'rew': self._to_tensor(self.rew_buf[idxs]),
            'next_global_state': self._to_tensor(self.next_global_state_buf[idxs]),
            'next_obs': self._to_tensor(self.next_obs_buf[idxs]),
            'done': self._to_tensor(self.done_buf[idxs])
        }
        
        return data

    def _to_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)
    
    def __len__(self):
        return self.size