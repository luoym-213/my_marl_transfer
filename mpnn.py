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

        # ==================== 基础配置 ====================
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
        num_actions = action_space.n

        # ==================== 模块化网络 ====================
        self.modules_dict = nn.ModuleDict()
        
        # 1. 底层策略网络
        self.modules_dict['low_level'] = self._build_low_level_modules(action_space)
        
        # 2. 高层策略网络
        self.modules_dict['high_level'] = self._build_high_level_modules()
        
        # 3. 高层 Critic
        self.modules_dict['high_critic'] = self._build_high_critic_modules()

        # ==================== 其他属性 ====================
        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)
        self.attn_mat = np.ones((num_agents, num_agents))
        self.dropout_mask = None     
        

        # self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
        #                                 self.nonlin(inplace=True),
        #                                 nn.Linear(self.h_dim,1))

        # self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
        #                                  self.nonlin(inplace=True))

        # self.low_agent_encoder = nn.Sequential(nn.Linear(self.low_level_input, self.h_dim),
        #                                       self.nonlin(inplace=True))
        
        # ==================== 代办 ====================
        self.dist = Categorical(self.h_dim,num_actions)

    # ==================== 模块构建函数 ====================
    def _build_low_level_modules(self, action_space):
        """构建底层策略网络"""
        num_actions = action_space.n
        low_level = nn.ModuleDict({
            'encoder': nn.Sequential(
                nn.Linear(self.low_level_input, self.h_dim),
                self.nonlin(inplace=True)
            ),
            'value_head': nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                self.nonlin(inplace=True),
                nn.Linear(self.h_dim, 1)
            ),
            'policy_head': nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                self.nonlin(inplace=True)
            ),
            'dist': Categorical(self.h_dim, num_actions)
        })
        
        return low_level

    def _build_high_level_modules(self):
        """构建高层策略网络（Actor）"""
        high_level = nn.ModuleDict({
            # 地图编码器
            'map_conv1': nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU()
            ),
            'map_conv2': nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            'map_conv3': nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
                nn.ReLU()
            ),
            
            # 向量编码器
            'vec_mlp': nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            ),
            
            # 决策头
            'decision_head': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ),
            
            # 探索点解码器
            'decoder_fuse': nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=1),
                nn.ReLU()
            ),
            'decoder_up1': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'decoder_up2': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'decoder_out': nn.Sequential(
                nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),
                nn.Conv2d(16, 1, kernel_size=1)
            )
        })
        
        return high_level

    def _build_high_critic_modules(self):
        """构建高层 Critic"""
        high_critic = nn.ModuleDict({
            'map_backbone': nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
                nn.ReLU()
            ),
            'map_compress': nn.Sequential(
                nn.Linear(64 * 6 * 6, 256),
                nn.ReLU()
            ),
            'vec_encoder': nn.Sequential(
                nn.Linear(self.num_agents * 2 + self.num_agents * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            ),
            'fusion_layer': nn.Sequential(
                nn.Linear(256 + 128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
        })
        
        # 每个智能体的独立输出头
        high_critic['value_heads'] = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(self.num_agents)
        ])
        
        return high_critic

    # ==================== 参数管理接口 ====================
    
    def get_module_params(self, module_name):
        """
        获取指定模块的参数
        
        Args:
            module_name: 'shared', 'low_level', 'high_level', 'high_critic'
        
        Returns:
            list of parameters
        """
        if module_name not in self.modules_dict:
            raise ValueError(f"Module '{module_name}' not found! Available: {list(self.modules_dict.keys())}")
        
        return list(self.modules_dict[module_name].parameters())

    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        for param in self.get_module_params(module_name):
            param.requires_grad = False
        print(f"✅ Module '{module_name}' frozen")

    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        for param in self.get_module_params(module_name):
            param.requires_grad = True
        print(f"✅ Module '{module_name}' unfrozen")

    def save_module_checkpoint(self, module_name, path):
        """
        保存指定模块的参数
        
        Args:
            module_name: 模块名称
            path: 保存路径
        """
        if module_name not in self.modules_dict:
            raise ValueError(f"Module '{module_name}' not found!")
        
        checkpoint = {
            'module_name': module_name,
            'state_dict': self.modules_dict[module_name].state_dict(),
            'config': {
                'num_agents': self.num_agents,
                'num_entities': self.num_entities,
                'hidden_dim': self.h_dim,
                'embed_dim': self.embed_dim
            }
        }
        
        torch.save(checkpoint, path)
        print(f"✅ Saved '{module_name}' checkpoint to {path}")

    def load_module_checkpoint(self, module_name, path, strict=True, freeze=False):
        """
        加载指定模块的参数
        
        Args:
            module_name: 模块名称
            path: checkpoint 路径
            strict: 是否严格匹配参数
            freeze: 是否加载后冻结
        
        Returns:
            missing_keys, unexpected_keys
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # 验证配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            if config.get('num_agents') != self.num_agents:
                print(f"⚠️ Warning: num_agents mismatch! "
                      f"Checkpoint: {config['num_agents']}, Current: {self.num_agents}")
        
        # 加载参数
        missing, unexpected = self.modules_dict[module_name].load_state_dict(
            checkpoint['state_dict'], 
            strict=strict
        )
        
        if freeze:
            self.freeze_module(module_name)
        
        print(f"✅ Loaded '{module_name}' checkpoint from {path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        
        return missing, unexpected

    def save_all_modules(self, save_dir):
        """保存所有模块到指定目录"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for module_name in self.modules_dict.keys():
            save_path = os.path.join(save_dir, f"{module_name}.pth")
            self.save_module_checkpoint(module_name, save_path)

    def load_pretrained_low_level(self, path, freeze=True):
        """
        智能加载函数：支持加载 '模块化Checkpoint' 或 '完整训练Checkpoint'
        """
        print(f"🔄 Loading low-level params from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        low_level_state_dict = {}
        
        # === 情况 A: 这是一个模块化 Checkpoint ===
        if 'state_dict' in checkpoint:
            print("  Type: Module Checkpoint")
            low_level_state_dict = checkpoint['state_dict']
            
        # === 情况 B: 这是一个完整训练 Checkpoint ===
        elif 'models' in checkpoint:
            print("  Type: Full Training Checkpoint (extracting params...)")
            full_state_dict = checkpoint['models'][0]
            
            # ⭐ 关键修改：明确底层网络的键名前缀
            # 旧代码中，底层网络的键名应该是 'low_agent_encoder.*', 'value_head.*' 等
            target_keys = [
                'low_agent_encoder',  # ← 这是底层编码器的真正名字
                'value_head',
                'policy_head',
                'dist'
            ]
            
            for key, value in full_state_dict.items():
                # 去除可能的 'modules_dict.low_level.' 前缀（如果是新版代码保存的）
                clean_key = key.replace('modules_dict.low_level.', '')
                
                # 检查是否属于底层网络（必须完整匹配前缀）
                if any(clean_key.startswith(prefix) for prefix in target_keys):
                    # ⭐ 如果是旧代码，需要将 'low_agent_encoder' 映射为 'encoder'
                    # 因为新代码中底层模块内部的名字是 'encoder'
                    final_key = clean_key.replace('low_agent_encoder', 'encoder')
                    low_level_state_dict[final_key] = value
                    
        else:
            raise ValueError(f"Unknown checkpoint format! Keys found: {list(checkpoint.keys())}")

        # 加载参数
        missing, unexpected = self.modules_dict['low_level'].load_state_dict(
            low_level_state_dict, 
            strict=False 
        )
        
        if freeze:
            self.freeze_module('low_level')
            
        print(f"✅ Low-level loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  ⚠️ Missing: {missing}")
        if unexpected:
            print(f"  ⚠️ Unexpected: {unexpected}")
        
        return missing, unexpected
        """
        便捷函数：加载预训练的底层网络
        
        Args:
            path: checkpoint 路径
            freeze: 是否冻结参数
        """
        # return self.load_module_checkpoint('low_level', path, strict=False, freeze=freeze)

    def get_trainable_params_by_modules(self, module_names, learning_rates=None):
        """
        获取多个模块的参数组（用于优化器）
        
        Args:
            module_names: 模块名称列表
            learning_rates: 对应的学习率列表（可选）
        
        Returns:
            param_groups for optimizer
        
        Example:
            >>> param_groups = model.get_trainable_params_by_modules(
            ...     ['high_level', 'high_critic', 'low_level'],
            ...     [3e-4, 3e-4, 1e-5]  # 底层使用更小的学习率
            ... )
            >>> optimizer = torch.optim.Adam(param_groups)
        """
        param_groups = []
        
        if learning_rates is None:
            learning_rates = [None] * len(module_names)
        
        for module_name, lr in zip(module_names, learning_rates):
            params = self.get_module_params(module_name)
            if lr is not None:
                param_groups.append({'params': params, 'lr': lr})
            else:
                param_groups.append({'params': params})
        
        return param_groups

    # ==================== 前向传播接口（保持兼容）====================
    
    @property
    def low_agent_encoder(self):
        return self.modules_dict['low_level']['encoder']
    
    @property
    def value_head(self):
        return self.modules_dict['low_level']['value_head']
    
    @property
    def policy_head(self):
        return self.modules_dict['low_level']['policy_head']
    
    @property
    def dist(self):
        return self.modules_dict['low_level']['dist']
    
    @property
    def map_conv1(self):
        return self.modules_dict['high_level']['map_conv1']
    
    @property
    def map_conv2(self):
        return self.modules_dict['high_level']['map_conv2']
    
    @property
    def map_conv3(self):
        return self.modules_dict['high_level']['map_conv3']
    
    @property
    def vec_mlp(self):
        return self.modules_dict['high_level']['vec_mlp']
    
    @property
    def decision_head(self):
        return self.modules_dict['high_level']['decision_head']
    
    @property
    def decoder_fuse(self):
        return self.modules_dict['high_level']['decoder_fuse']
    
    @property
    def decoder_up1(self):
        return self.modules_dict['high_level']['decoder_up1']
    
    @property
    def decoder_up2(self):
        return self.modules_dict['high_level']['decoder_up2']
    
    @property
    def decoder_out(self):
        return self.modules_dict['high_level']['decoder_out']
    
    @property
    def critic_map_backbone(self):
        return self.modules_dict['high_critic']['map_backbone']
    
    @property
    def critic_map_compress(self):
        return self.modules_dict['high_critic']['map_compress']
    
    @property
    def critic_vec_encoder(self):
        return self.modules_dict['high_critic']['vec_encoder']
    
    @property
    def critic_fusion_layer(self):
        return self.modules_dict['high_critic']['fusion_layer']
    
    @property
    def critic_value_out_heads(self):
        return self.modules_dict['high_critic']['value_heads']
    
    @property
    def critic_map_flat_dim(self):
        return 64 * 6 * 6
    
    @property
    def critic_vec_input_dim(self):
        return self.num_agents * 2 + self.num_agents * 2

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
        y_coords = torch.div(flat_idx, 100, rounding_mode='floor')  # 向下取整（推荐用于坐标计算）
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
        dist_decision = TorchCategorical(probs=decision_probs)
        
        # 计算给定动作的 log_prob
        decision_log_probs = dist_decision.log_prob(tasks.squeeze(-1))  # [batch]
        
        # 计算每个样本的熵
        decision_entropy = dist_decision.entropy()  # [batch]

        # =====================================================
        # 5. 计算探索头的 log_probs 和 entropy
        # =====================================================
        heatmap_probs = F.softmax(heatmap_logits, dim=-1)  # [batch, 10000]
        dist_map = TorchCategorical(probs=heatmap_probs)
        
        # 将 goals [batch, 2] 转换为 flat_idx [batch]
        goals_x = goals[:, 0].long()  # [batch]
        goals_y = goals[:, 1].long()  # [batch]
        flat_idx = goals_y * 100 + goals_x  # [batch]
        
        # 计算给定导航点的 log_prob
        map_log_probs = dist_map.log_prob(flat_idx)  # [batch]
        
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
