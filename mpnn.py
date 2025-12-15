import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from torch.distributions import Categorical as TorchCategorical

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

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None        

        # ================================ Critic Start ==================================== 
        # ==========================================
        # 1. 全局地图编码器 (Global Map Encoder)
        # 输入: [B, 4, 100, 100] -> 输出: [B, 256]
        # ==========================================
        self.critic_map_backbone = nn.Sequential(
            # L1: 100 -> 50
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # L2: 50 -> 25
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # L3: 25 -> 13 (padding=1)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # L4: 13 -> 6 (padding=0, 丢弃最外圈无用信息，进一步压缩)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), 
            nn.ReLU()
        )
        
        # 计算 Flatten 后的维度: 64 * 6 * 6 = 2304
        self.critic_map_flat_dim = 64 * 6 * 6
        
        # 降维层：将展平后的图像特征压缩，方便与向量融合
        self.critic_map_compress = nn.Sequential(
            nn.Linear(self.critic_map_flat_dim, 256),
            nn.ReLU()
        )

        # ==========================================
        # 2. 全局向量编码器 (Global Vector Encoder)
        # 输入: [B, N_vec] -> 输出: [B, 128]
        # ==========================================
        # 向量维度: N个智能体(x,y) + M个宝藏(x,y)
        self.critic_vec_input_dim = num_agents * 2 + num_agents * 2
        
        self.critic_vec_encoder = nn.Sequential(
            nn.Linear(self.critic_vec_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # ==========================================
        # 3. 融合与价值头 (Fusion & Value Head)
        # 输入: 256(Map) + 128(Vec) = 384
        # 输出: [B, num_agents]
        # ==========================================
        self.critic_fusion_layer = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 为每个智能体创建独立的输出头
        self.critic_value_out_heads = nn.ModuleList([
            nn.Linear(128, 1) 
            for _ in range(self.num_agents)
        ])

        # ================================ Critic End ==================================== 

        # ================================ Actor Start ====================================
        # ====================================================================
        # 1. 地图流编码器 (Map Encoder)
        # 输入: [Batch, 4, 100, 100] -> 输出: [Batch, 64, 12, 12]
        # ====================================================================
        self.map_conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2), # 100 -> 50
            nn.ReLU()
        )
        self.map_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 50 -> 25
            nn.ReLU()
        )
        self.map_conv3 = nn.Sequential(
            # Padding=0 是关键，切掉边缘，从 25x25 变成 12x12
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0), # 25 -> 12
            nn.ReLU()
        )

        # ====================================================================
        # 2. 向量流编码器 (Vector Encoder)
        # 输入: [Batch, 5] -> 输出: [Batch, 64]
        # ====================================================================
        self.vec_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # ====================================================================
        # 3. 决策头 (Decision Head)
        # 输入: h_shared [128] -> 输出: Logits [2] (Explore vs Collect)
        # ====================================================================
        self.decision_head = nn.Sequential(
            nn.Linear(128, 64), # 128 = 64(Map) + 64(Vec)
            nn.ReLU(),
            nn.Linear(64, 2)    # 输出 Explore, Collect 的 Logits
        )

        # ====================================================================
        # 4. 探索点生成头 (Waypoint Head - Decoder)
        # 输入: F_spatial + h_shared -> 输出: Heatmap [1, 100, 100]
        # ====================================================================
        
        # 特征融合层：将拼接后的 192 维特征降维融合
        self.decoder_fuse = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1), # 192 = 64(Spatial) + 128(Broadcasted Shared)
            nn.ReLU()
        )
        
        # 上采样层 1: 12x12 -> 24x24
        self.decoder_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 上采样层 2: 24x24 -> 48x48
        self.decoder_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 上采样层 3: 48x48 -> 100x100 (直接指定 size)
        self.decoder_out = nn.Sequential(
            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1) # 输出单通道 Logits
            # 注意：这里没有 Activation，因为后续要做 Softmax 和 Masking
        )


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

    def forward(self, inp, state, mask=None):
        # 保存当前的隐藏状态
        if state is not None:
            self.hidden_state = state
            
        x = self._fwd(inp)
        # 返回更新后的隐藏状态
        return x, self.hidden_state.clone()

    def high_level_forward(self, vec_inp, map_inp):
        """
        map_input: [Batch, 4, 100, 100] 
                   Channel 0: Belief Map 
                   Channel 1: Entropy Map
                   Channel 2: Voronoi Mask (0/1) 
                   Channel 3: Distance Field
        vec_input: [Batch, 5]
        """
        batch_size = map_inp.size(0)

        # ---------------------------
        # A. 提取特征 (Backbone)
        # ---------------------------
        
        # 1. 空间特征 F_spatial: [B, 64, 12, 12]
        f_spatial = self.map_conv1(map_inp)
        f_spatial = self.map_conv2(f_spatial)
        f_spatial = self.map_conv3(f_spatial)

        # 2. 向量特征 f_vec: [B, 64]
        f_vec = self.vec_mlp(vec_inp)

        # ---------------------------
        # B. 特征融合 (Masked Pooling)
        # ---------------------------
        
        # 1. 取出原始 Voronoi Mask (假设是第2个通道) -> [B, 2, 100, 100]
        voronoi_mask_raw = map_inp[:, 2:3, :, :]

        # 2. 下采样 Mask 到 12x12，模式为 nearest (保持 0/1 硬边界)
        mask_down = F.interpolate(voronoi_mask_raw, size=(12, 12), mode='nearest')
        
        # 3. Masked Global Average Pooling
        # 只对 Voronoi 区域内的特征求和
        masked_features = f_spatial * mask_down 
        sum_features = torch.sum(masked_features, dim=(2, 3)) # [B, 64]
        
        # 计算区域面积 (像素数)，加 epsilon 防止除以 0
        area = torch.sum(mask_down, dim=(2, 3)) + 1e-5 # [B, 1]
        
        # 得到局部区域的平均特征向量 v_map: [B, 64]
        v_map = sum_features / area 

        # 4. 生成共享向量 h_shared: [B, 128]
        h_shared = torch.cat([v_map, f_vec], dim=1)

        # ---------------------------
        # C. 决策头 (Decision Head)
        # ---------------------------
        
        # 输出 [Explore, Collect] 的 Logits: [B, 2]
        decision_logits = self.decision_head(h_shared)

        # ---------------------------
        # D. 探索点生成头 (Waypoint Head)
        # ---------------------------
        
        # 1. 广播 h_shared 到空间尺寸: [B, 128, 12, 12]
        h_shared_expanded = h_shared.view(batch_size, 128, 1, 1).expand(-1, -1, 12, 12)
        
        # 2. 拼接空间特征与全局特征: [B, 192, 12, 12]
        decoder_in = torch.cat([f_spatial, h_shared_expanded], dim=1)
        
        # 3. 解码生成热力图 Logits: [B, 1, 100, 100]
        x = self.decoder_fuse(decoder_in)
        x = self.decoder_up1(x)
        x = self.decoder_up2(x)
        heatmap_logits = self.decoder_out(x)

        # ---------------------------
        # E. 后处理 (Masking)
        # ---------------------------
        
        # 关键步骤：再次使用原始 Voronoi Mask
        # 将非势力范围内的 Logits 设为负无穷
        # mask == 0 的地方填入 -1e9
        heatmap_logits = heatmap_logits.masked_fill(voronoi_mask_raw == 0, -1e9)
        
        # 展平以便后续采样: [B, 10000]
        flat_heatmap_logits = heatmap_logits.view(batch_size, -1)

        return decision_logits, flat_heatmap_logits

    def get_high_level_goal(self, vec_inp, map_inp, deterministic=False):
        """
        采样动作的辅助函数 (用于 Rollout)
        """
        decision_logits, heatmap_logits = self.high_level_forward(vec_inp, map_inp)

        # ============================================================
        # 基于 B_candidate 的动作掩码 (Action Masking)
        # 向量流结构: <x_pos, y_pos, B_candidate, x_target, y_target>
        # B_candidate 位于索引 2
        b_candidate = vec_inp[:, 2] # Shape: [Batch]

        # 为了不破坏计算图（如果是训练过程），建议 clone 一份 logits
        # 如果只是推理 rollout，不 clone 也可以，但 clone 是好习惯
        masked_decision_logits = decision_logits.clone()

        # 找到没有候选点的样本索引 (B_candidate == 0)
        # 注意：浮点数比较建议用 < 0.5 或者 isclose，这里假设输入是严格的 0/1
        mask_no_candidate = (b_candidate < 0.5) 

        # 将这些样本的 COLLECT 动作 (索引 1) 的 Logit 设为负无穷
        # -1e9 在 Softmax 后会变成 0
        masked_decision_logits[mask_no_candidate, 1] = -1e9

        # ============================================================
        # 采样动作并计算 log_probs（关键修改）
        # ============================================================
    
        # 1. 决策动作
        decision_probs = F.softmax(masked_decision_logits, dim=-1)
        dist_mode = TorchCategorical(probs=decision_probs)
    
        if deterministic:
            action_mode = torch.argmax(decision_probs, dim=-1)
        else:
            action_mode = dist_mode.sample().squeeze(-1)    # [batch]



        # ✅ 计算决策动作的 log_prob
        decision_log_prob = dist_mode.log_prob(action_mode)  # [Batch]
    
        # 2. 导航点动作
        heatmap_probs = F.softmax(heatmap_logits, dim=-1)   # [Batch, 10000]
        dist_map = TorchCategorical(probs=heatmap_probs)

        if deterministic:
            flat_idx = torch.argmax(heatmap_probs, dim=-1)
        else:
            flat_idx = dist_map.sample().squeeze(-1)    # [batch]
    
        # ✅ 计算导航点动作的 log_prob
        map_log_prob = dist_map.log_prob(flat_idx)  # [Batch]
    
        # 3. 转换坐标
        y_coords = flat_idx // 100
        x_coords = flat_idx % 100
        if x_coords.dim() > 1:
            x_coords = x_coords.squeeze(-1)  # 移除最后一维
        if y_coords.dim() > 1:
            y_coords = y_coords.squeeze(-1)

        # 4. 根据决策调整导航点
        target_x = vec_inp[:, 3]
        target_y = vec_inp[:, 4]
        collect_mask = (action_mode == 1)  # [Batch]
        x_coords = torch.where(collect_mask, target_x.long(), x_coords)
        y_coords = torch.where(collect_mask, target_y.long(), y_coords)
        waypoints = torch.stack([x_coords, y_coords], dim=1).float()  # [Batch, 2]
        
        return {
            "action_modes": action_mode.unsqueeze(-1),              # [Batch, 1]
            "waypoints": waypoints,                   # [Batch,2]
            "decision_log_probs": decision_log_prob.unsqueeze(-1),  # ✅ [Batch, 1] 标量
            "map_log_probs": map_log_prob.unsqueeze(-1)            # ✅ [Batch, 1] 标量
        }
    
    def get_high_value(self, map_inp, vec_inp):
        # map_inp: [num_processes, 4, H, W]
        # vec_inp: [num_processes, num_agents*2 + num_landmarks*2]
        # 返回： values: [num_processes, num_agents]
        batch_size = map_inp.size(0)

        # 全局地图编码器
        f_map = self.critic_map_backbone(map_inp)  # [B, 64, 6, 6]
        f_map_flat = f_map.view(batch_size, -1)  # [B, 64*6*6]
        f_map_compress = self.critic_map_compress(f_map_flat)  # [B, 256]

        # 结合全局地图特征和局部特征
        f_vec = self.critic_vec_encoder(vec_inp)  # [B, 128]
        f_fuse = torch.cat([f_map_compress, f_vec], dim=1)  # [B, 256 + 128]

        fused = self.critic_fusion_layer(f_fuse)    # [B, 128]

        # 为每个智能体计算独立的价值
        values = []
        for agent_idx in range(self.num_agents):
            value = self.critic_value_out_heads[agent_idx](fused)  # [B, 1]
            values.append(value)
        values = torch.cat(values, dim=1)  # [B, num_agents]

        return values

    def _low_value(self, x):
        return self.value_head(x) # h_dim -> h_dim -> 1

    def _low_policy(self, x): # h_dim -> h_dim
        return self.policy_head(x)
    
    def vec_inp_generator(self, env_state, detected_map):
        # 生成智能体向量流，得到vec_inp [num_agents, 5]，<x_pos, y_pos, B_candidate, x_target, y_target>
        # 使用匈牙利算法为每个智能体分配最近的已发现目标点作为目标位置，已发现目标点可能小于智能体数量
        # 如果没分配到，默认<x_pos, y_pos, 0, 0, 0>，否则<x_pos, y_pos, 1, x_target, y_target>
        # 提取智能体当前位置, [num_agents, 2]
        agents_pos = env_state[self.num_agents * 2:self.num_agents * 4].view(self.num_agents, 2) 

        # 提取已发现目标点位置, [num_detected, 2]
        detected_pos = detected_map.view(-1, 2)

        # 计算智能体到已发现目标点的距离
        cost_matrix = torch.cdist(agents_pos, detected_pos, p=2)  # [num_agents, num_detected]

        # 使用匈牙利算法进行最优分配
        cost_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # 生成最终的向量流，首先用智能体当前位置初始化
        vec_inp = torch.zeros((self.num_agents, 5), device=env_state.device)
        vec_inp[:, :2] = agents_pos

        # 对于被分配到目标的智能体，更新其备选点标记和目标位置
        for agent_idx, target_idx in zip(row_ind, col_ind):
            vec_inp[agent_idx, 2] = 1.0  # 备选点标记
            vec_inp[agent_idx, 3:5] = detected_pos[target_idx]  # 目标位置

        return vec_inp

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
    

    def evaluate_high_actions(self, env_states, map_obs, vec_obs, critic_maps, goals, tasks, agent_ids):
        """
        Input:
            env_states: [batch, env_dim]
            map_obs: [batch, 4, H, W]
            vec_obs: [batch, 5] 
            agent_ids: [batch, 1]
        评估高层动作
        Returns:
            high_values: [batch, 1]
            decision_log_probs: [batch, 1]
            map_log_probs: [batch, 1]
            decision_entropy: [batch] ← 改为每个样本的熵
            waypoint_entropy: [batch] ← 改为每个样本的熵
        """
        batch_size = map_obs.size(0)
        num_agents = self.num_agents

        # =====================================================
        # 1. Critic: 计算所有智能体的价值 [batch, num_agents]
        # =====================================================
        critic_vec = env_states[:, 2*num_agents:]  # 提取全局向量信息
        all_values = self.get_high_value(critic_maps, critic_vec)  # [batch, num_agents]
        
        # ⭐ 根据 agent_ids 选择对应的价值
        agent_ids_expanded = agent_ids.unsqueeze(-1)  # [batch, 1]
        high_values = torch.gather(all_values, dim=1, index=agent_ids_expanded)  # [batch, 1]

        # =====================================================
        # 2. Actor: 前向传播获取 logits
        # =====================================================
        decision_logits, heatmap_logits = self.high_level_forward(vec_obs, map_obs)
        # decision_logits: [batch, 2]
        # heatmap_logits: [batch, 10000]

        # =====================================================
        # 3. 应用动作掩码 (Action Masking)
        # =====================================================
        b_candidate = vec_obs[:, 2]  # [batch]
        masked_decision_logits = decision_logits.clone()
        
        # 将没有候选点的样本的 COLLECT 动作设为负无穷
        mask_no_candidate = (b_candidate < 0.5)
        masked_decision_logits[mask_no_candidate, 1] = -1e9

        # =====================================================
        # 4. 计算决策头的 log_probs 和 entropy
        # =====================================================
        decision_probs = F.softmax(masked_decision_logits, dim=-1)  # [batch, 2]
        dist_decision = Categorical(decision_probs)
        
        # 计算给定动作的 log_prob
        decision_log_probs = dist_decision.log_probs(tasks.squeeze(-1)).unsqueeze(-1)  # [batch, 1]
        
        # 计算每个样本的熵
        decision_entropy = dist_decision.entropy()  # [batch]

        # =====================================================
        # 5. 计算探索头的 log_probs 和 entropy
        # =====================================================
        heatmap_probs = F.softmax(heatmap_logits, dim=-1)  # [batch, 10000]
        dist_map = Categorical(heatmap_probs)
        
        # 将 goals [batch, 2] 转换为 flat_idx [batch]
        goals_x = goals[:, 0].long()  # [batch]
        goals_y = goals[:, 1].long()  # [batch]
        flat_idx = goals_y * 100 + goals_x  # [batch]
        
        # 计算给定导航点的 log_prob
        map_log_probs = dist_map.log_probs(flat_idx).unsqueeze(-1)  # [batch, 1]
        
        # 计算每个样本的熵
        waypoint_entropy = dist_map.entropy()  # [batch]

        return (high_values, decision_log_probs, map_log_probs, 
                decision_entropy, waypoint_entropy)
    
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
