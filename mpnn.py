import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment  # 匈牙利算法

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MPNN(nn.Module):
    def __init__(self, action_space, num_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=3, mask_dist=None, mask_obs_dist=None, entity_mp=False, is_recurrent=True):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents # number of agents
        self.num_entities = num_entities # number of entities
        self.low_level_input = 2 + 2*num_agents # low level input size: agent pos + all agents pos
        self.K = 3 # message passing rounds
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.is_recurrent = is_recurrent
        self.mask_dist = mask_dist
        self.mask_obs_dist = mask_obs_dist
        self.input_size = input_size # 这里是agengt自身速度位置（4）
        self.entity_mp = entity_mp
        # this index must be from the beginning of observation vector
        self.pos_index = pos_index
        # task generation parameters
        self.task_dim = 2
        self.h_dim2 = self.h_dim // 2 # 64

        self.encoder = nn.Sequential(nn.Linear(self.input_size,self.h_dim),
                                     self.nonlin(inplace=True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                    self.nonlin(inplace=True))

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim,1))

        self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True))
        
        self.agent_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                     self.nonlin(inplace=True))
        
        self.agent_messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)

        self.agent_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                          self.nonlin(inplace=True))

        self.low_agent_encoder = nn.Sequential(nn.Linear(self.low_level_input, self.h_dim),
                                              self.nonlin(inplace=True))

        if self.entity_mp:
            self.entity_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                                self.nonlin(inplace=True))
            
            self.entity_messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)
            
            self.entity_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                               self.nonlin(inplace=True))
        
        num_actions = action_space.n
        self.dist = Categorical(self.h_dim,num_actions)

        # 添加GRU层
        self.gru = nn.GRUCell(self.h_dim, self.h_dim) 

        # TaskGeneration module
        self.task = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),self.nonlin(inplace=True), # 128 -> 128
            nn.Linear(self.h_dim, self.h_dim2),self.nonlin(inplace=True), # 128 -> 64
            nn.Linear(self.h_dim2, self.task_dim)    # 64 -> task_dim
        )

        self.task_encoder = nn.Sequential(nn.Linear(self.task_dim, self.h_dim2),self.nonlin(inplace=True)) # task_dim -> 64

        # 拼接融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim2, self.h_dim),
            self.nonlin(inplace=True)
        )

        # Centralized Critic
        self.critic_shared = nn.Sequential(
            nn.Linear(self.num_agents * 6 + self.task_dim * self.num_agents, self.h_dim * 2),
            self.nonlin(inplace=True),
            nn.Linear(self.h_dim * 2, self.h_dim * 2),
            self.nonlin(inplace=True)
        )

        # 为每个智能体创建独立的输出头
        self.critic_heads = nn.ModuleList([
            nn.Linear(self.h_dim * 2, 1) 
            for _ in range(self.num_agents)
        ])

        # self.critic_value_head = nn.Sequential(nn.Linear(self.num_agents * 6 + self.task_dim * self.num_agents, self.h_dim * 2), # env_state + task
        #                             self.nonlin(inplace=True),
        #                             nn.Linear(self.h_dim * 2, self.h_dim * 2),
        #                             self.nonlin(inplace=True),
        #                             nn.Linear(self.h_dim * 2, 1))

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None

    def calculate_mask(self, inp):
        # inp is batch_size x self.input_size where batch_size is num_processes*num_agents
        
        pos = inp[:, self.pos_index:self.pos_index+2]
        bsz = inp.size(0)//self.num_agents
        pos_obs = inp[:, self.pos_index:self.pos_index+2]
        bsz_obs = inp.size(0)//self.num_agents
        mask = torch.full(size=(bsz,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        mask_agents = torch.full(size=(bsz_obs,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        
        if self.mask_dist is not None and self.mask_dist > 0: 
            for i in range(1,self.num_agents):
                shifted = torch.roll(pos,-bsz*i,0)
                dists = torch.norm(pos-shifted,dim=1)
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:,x,(x+i)%self.num_agents].copy_(restrict[bsz*x:bsz*(x+1)])
        
        elif self.mask_dist is not None and self.mask_dist == -10:
           if self.dropout_mask is None or bsz!=self.dropout_mask.shape[0] or np.random.random_sample() < 0.1: # sample new dropout mask
               temp = torch.rand(mask.size()) > 0.85
               temp.diagonal(dim1=1,dim2=2).fill_(0)
               self.dropout_mask = (temp+temp.transpose(1,2))!=0
           mask.copy_(self.dropout_mask)

        if self.mask_obs_dist is not None and self.mask_obs_dist > 0: 
            for j in range(1,self.num_agents):
                shifted_obs = torch.roll(pos_obs,-bsz_obs*j,0)
                dists_obs = torch.norm(pos_obs-shifted_obs,dim=1)
                restrict_obs = dists_obs > self.mask_obs_dist
                for y in range(self.num_agents):
                    mask_agents[:,y,(y+j)%self.num_agents].copy_(restrict_obs[bsz_obs*y:bsz_obs*(y+1)])
        mask_agents = mask_agents.masked_select(~torch.eye(self.num_agents, self.num_agents, device=mask_agents.device, dtype=torch.bool)).view(bsz_obs, self.num_agents, self.num_agents-1)
        
        return mask,mask_agents             

    def calculate_mask_entity(self, inp):
        # inp: landmark's positon, [num_processes * num_agents, num_agents*2]
        bsz = inp.size(0)//self.num_agents  # num_processes
        dists = torch.norm(inp.contiguous().view(bsz * self.num_agents, self.num_agents, 2), p=2, dim=2) # [bsz*num_agents, dis_landmark]
        #restrict = dists > self.mask_obs_dist # [bsz*num_agents, dis_landmark]
        dists_reshape = dists.contiguous().view(self.num_agents, bsz, self.num_agents).permute(1, 0, 2) # [bsz, self.num_agents,self.num_agents]

        # 全通信下只需计算每个地标距离智能体的最小距离
        min_dists = dists_reshape.min(dim=1, keepdim=True)[0]  # [bsz, 1, num_landmarks]
        #mask =restrict.contiguous().view(self.num_agents, bsz, self.num_agents).permute(1, 0, 2) # [bsz, self.num_agents,self.num_agents]
        mask_global = (min_dists > self.mask_obs_dist).expand(bsz, self.num_agents, self.num_agents)
        return mask_global

    def _fwd(self, inp, state=None):
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
        # state should be (batch_size, hidden_size)
        agent_inp = inp[:,:self.input_size]
        mask, mask_agents = self.calculate_mask(agent_inp) # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted         

        h = self.encoder(agent_inp) # should be (batch_size,self.h_dim)
        '''
        other_agent_inp = inp[:,self.input_size+self.num_entities*2:] # x,y relative pos of agents wrt agents
        ha = self.agent_encoder(other_agent_inp.contiguous().view(-1,2)).view(-1,self.num_agents-1,self.h_dim) # [num_agents*num_processes, num_agents-1, 128]
        #print("ha shape: ", ha.shape) 
        agent_message,agent_attn = self.agent_messages(h.unsqueeze(1),ha,mask=mask_agents,return_attn = True)
        h = self.agent_update(torch.cat((h,agent_message.squeeze(1)),1)) # should be (batch_size,self.h_dim)
        #print("h shape: ", h.shape)
        '''
        if self.entity_mp:
            landmark_inp = inp[:,self.input_size:self.input_size+self.num_entities*2] # x,y pos of landmarks wrt agents
            # should be (batch_size,self.num_entities,self.h_dim)
            # compute entity mask
            mask_entity = self.calculate_mask_entity(landmark_inp)
            he = self.entity_encoder(landmark_inp.contiguous().view(-1,2)).view(-1,self.num_entities,self.h_dim)
            # entity_message = self.entity_messages(h.unsqueeze(1),he).squeeze(1) # should be (batch_size,self.h_dim)
            entity_message,entity_attn = self.entity_messages(h.unsqueeze(1),he,mask=mask_entity,return_attn=True) # should be (batch_size,self.h_dim)
            h = self.entity_update(torch.cat((h,entity_message.squeeze(1)),1)) # should be (batch_size,self.h_dim)

        # h = h.view(self.num_agents,-1,self.h_dim).transpose(0,1) # should be (batch_size/N,N,self.h_dim)
        
        # 使用GRU处理输入
        new_state = self.gru(h, state)
        
        # 将新的隐藏状态重新塑形回 (batch_size_per_agent, num_agents, hidden_dim)
        h = new_state.view(self.num_agents,-1,self.h_dim).transpose(0,1) # should be (batch_size/N,N,self.h_dim)

        for k in range(self.K):
            m, attn = self.messages(h, mask=mask, return_attn=True) # should be <batch_size/N,N,self.embed_dim>
            h = self.update(torch.cat((h,m),2)) # should be <batch_size/N,N,self.h_dim>
        h = h.transpose(0,1).contiguous().view(-1,self.h_dim)
        
        self.attn_mat = attn.squeeze().detach().cpu().numpy()
        # print("h shape: ", h.shape)
        return h, new_state # should be <batch_size, self.h_dim> again
    
    def taskG(self,h):
        # 智能体的任务生成器
        h_task_logit = self.task(h) # should be <batch_size, task_dim>
        h_task = F.one_hot(torch.argmax(h_task_logit,dim=1),num_classes=self.task_dim).float() # should be <batch_size, task_dim>

        return h_task_logit, h_task
    
    def _task_fwd(self, h, h_task):
        # 将来自cta或者智能体的任务生成编码，并与上层观测向量合成编码
        h_task_enc = self.task_encoder(h_task) # should be <batch_size, 64>

        # combine h and h_task_enc to [batch_size, self.h_dim + 64] and then to self.h_dim
        h = self.fusion(torch.cat((h, h_task_enc), dim=1)) # should be <batch_size, self.h_dim>
        
        return h

    def forward(self, inp, state, mask=None):
        # 保存当前的隐藏状态
        if state is not None:
            self.hidden_state = state
            
        x = self._fwd(inp)
        # 返回更新后的隐藏状态
        return x, self.hidden_state.clone()

    def _low_value(self, x):
        return self.value_head(x) # h_dim -> h_dim -> 1

    def _low_policy(self, x): # h_dim -> h_dim
        return self.policy_head(x)
    
    def CTA(self, env_state):
        # env_state should be (<batch_size, env_dim>)
        # batch_size = env_state.size(0) = num_agents * num_processes
        
        # 任务生成模块,匈牙利算法
        batch_size = env_state.size(0) // self.num_agents
        agents_pos = env_state[:batch_size, self.num_agents * 2:self.num_agents * 4].view(batch_size, self.num_agents, 2) # [batch_size, 3, 2]
        landmarks_pos = env_state[:batch_size, self.num_agents * 4:].view(batch_size, self.num_agents, 2) # [batch_size, 3, 2]

        ## 构造代价矩阵
        cost_matrix = torch.cdist(agents_pos, landmarks_pos, p=2) # [batch_size, 3, 3]

        # 逐批次应用匈牙利算法
        assignments = []
        dists_list = []
        
        for b in range(batch_size):
            # 对每个批次分别计算
            cost_b = cost_matrix[b].cpu().numpy()  # [3, 3]
            row_ind, col_ind = linear_sum_assignment(cost_b)

            dists_b = torch.tensor(cost_b[row_ind, col_ind], device=env_state.device)
            
            assignments.append(col_ind)
            dists_list.append(dists_b)
        
        # 合并所有批次的距离 [batch_size, num_agents]
        dists = torch.stack(dists_list, dim=0)  # [batch_size, 3]
        
        # 生成二分类 onehot 编码 [batch_size, num_agents, task_dim]
        task = torch.zeros((batch_size, self.num_agents, self.task_dim), 
                        dtype=torch.float32, device=env_state.device)

        # 使用张量操作：创建掩码并使用scatter
        close_mask = (dists < self.mask_obs_dist).unsqueeze(-1)  # [batch_size, num_agents, 1]
        
        # 对于距离近的，设置第1列为1；距离远的，设置第0列为1
        task[:, :, 1] = close_mask.squeeze(-1).float()      # 距离近：任务类型1
        task[:, :, 0] = (~close_mask.squeeze(-1)).float()   # 距离远：任务类型0
        
        task = task.transpose(0, 1).contiguous().view(batch_size * self.num_agents, self.task_dim)

        return task

    def high_level_act(self, search_map, detected_map, env_state):
        # search_map: (<agent_dim*2>) 熵加权质点图,[(mx1, my1), (mx2, my2), ...]
        # detected_map: (<detected_dim*2>) 已发现目标位置图，数量可能不同
        # env_state: (<env_dim>)

        # 提取智能体当前位置, [num_agents, 2]
        agents_pos = env_state[self.num_agents * 2:self.num_agents * 4].view(self.num_agents, 2) 

        # 已发现目标点数量
        num_detected = detected_map.size(0)
        # 初始化目标分配（默认使用搜索点）
        goals = search_map.clone()  # [num_agents, 2]
        # 如果有已发现的目标点
        if num_detected > 0:
            # 计算智能体到所有已发现目标点的距离矩阵
            cost_matrix = torch.cdist(agents_pos, detected_map.float(), p=2)  # [num_agents, num_detected]
            
            # 使用匈牙利算法进行最优分配
            cost_np = cost_matrix.cpu().numpy()
            
            # 如果目标点数量 >= 智能体数量，直接分配
            if num_detected >= self.num_agents:
                row_ind, col_ind = linear_sum_assignment(cost_np)
                # 所有智能体都分配到目标点
                for agent_idx, target_idx in zip(row_ind, col_ind):
                    goals[agent_idx] = detected_map[target_idx]
            else:
                # 如果目标点数量 < 智能体数量
                # 只分配部分智能体到目标点，其余使用搜索点
                row_ind, col_ind = linear_sum_assignment(cost_np)
                for agent_idx, target_idx in zip(row_ind, col_ind):
                    goals[agent_idx] = detected_map[target_idx]
                # 未分配的智能体保持使用原搜索点（已经在初始化时设置）

        # # 拼接所有批次的目标 [batch_size, num_agents, 2] -> [batch_size * num_agents, 2]
        # all_goals = torch.stack(all_goals, dim=0)  # [batch_size, num_agents, 2]
        # all_goals = all_goals.transpose(0, 1).contiguous().view(batch_size * self.num_agents, 2)

        return goals
    
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

    def critic_value(self, env_state, task, mask=None):
        # env_state should be (<batch_size, env_dim>)
        # task should be (<batch_size, task_dim>)
        # mask should be (<batch_size, 1>)

        # expand task to global_task
        batch_size = task.size(0)
        num_processes = batch_size // self.num_agents
        task_global = task.view(self.num_agents, num_processes, self.task_dim).transpose(0, 1).contiguous().view(num_processes, self.num_agents * self.task_dim)
        task = task_global.repeat_interleave(self.num_agents, dim=0) # should be <batch_size, task_dim * num_agents>

        x = torch.cat((env_state, task), dim=1) # should be <batch_size, env_dim + task_dim>
        # 通过共享主干
        share_features = self.critic_shared(x) # should be <batch_size, h_dim * 2>
        # 逐个计算后拼接
        values = []
        for i in range(self.num_agents):
            share_feature_i = share_features[i*num_processes:(i+1)*num_processes]
            value_i = self.critic_heads[i](share_feature_i)
            values.append(value_i)

        x = torch.cat(values, dim=0)
        return x

    def act(self, inp, state, env_state, mask=None, deterministic=False):
        """
        inp: [batch_size, dim_o]
        state: [batch_size, dim_h]
        env_state: [batch_size, env_dim]
        mask: [batch_size, 1], mask for actions
        """
        x, new_state = self._fwd(inp, state)
        # 这里需要把x存入buffer中，看是return比较好这里可以直接存
        tgnet_input = x  # 取前task_dim维度作为tgnet的输入 <batch_size, dim_h>

        # 训练阶段使用cta的任务
        cta_task = self.CTA(env_state) # should be <batch_size, task_dim>
        x = self._task_fwd(x, cta_task)  # 拼接任务向量

        # value 不再使用sigle agent的输出头，使用中心的critic
        # value = self._value(x)
        value = self.critic_value(env_state, cta_task, mask)

        dist = self.dist(self._policy(x))
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1,1)
        
        # 返回更新后的隐藏状态
        return value, action, action_log_probs, new_state, cta_task, tgnet_input

    def evaluate_actions(self, inp, state, mask, action, env_state, task=None): 
        x, new_state = self._fwd(inp, state)  # 修正：传递state参数
        x = self._task_fwd(x, task)

        # value = self._value(x) # [num_mini_batch, 1]
        value = self.critic_value(env_state, task, mask)
        dist = self.dist(self._policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean() # [num_mini_batch, 1] -> 1 scalar
        
        # 返回更新后的隐藏状态
        return value, action_log_probs, dist_entropy, new_state  # 修正：返回new_state而不是self.hidden_state.clone()
    
    def evaluate_low_actions(self, inp, goals, action):
        new_inp = self.data_processing_low_level(inp, goals)
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        dist = self.dist(self._low_policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy

    def get_value(self, inp, state, mask, env_state):
            
        x, new_state = self._fwd(inp, state)
        # 训练阶段使用cta的任务
        cta_task = self.CTA(env_state) # should be <batch_size, task_dim>
        x = self._task_fwd(x, cta_task)  # 添加任务生成模块
        # value = self._value(x)
        value = self.critic_value(env_state, cta_task, mask)
        return value
    
    def get_low_value(self, inp, goals):
        new_inp = self.data_processing_low_level(inp, goals)
        x = self.low_agent_encoder(new_inp)
        value = self._low_value(x)
        return value


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.contiguous().view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        if return_attn:
            return out, attn
        return out
