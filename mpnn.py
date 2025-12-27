import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from torch.distributions import Categorical as TorchCategorical
from planning.rrt_GNN import RRT_GNN, plan_batch

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
            
            # ===== 动态图推理模块 =====
            # Ego节点编码器 [5] -> [64]
            'ego_node_encoder': nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            ),
            
            # Explore节点编码器 [4] -> [64]
            'explore_node_encoder': nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            ),
            
            # Landmark节点编码器 [4] -> [64]
            'landmark_node_encoder': nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            ),
            
            # 边编码器 [3] -> [32]
            'edge_encoder': nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU()
            ),
            
            # 注意力投影层
            'q_proj': nn.Linear(64, 64),  # ego -> query
            'k_proj': nn.Linear(96, 64),  # node(64) + edge(32) -> key
            'v_proj': nn.Linear(96, 64),  # node(64) + edge(32) -> value
            
            # 节点选择头（统一对所有节点打分）
            'node_selection_head': nn.Linear(64, 1),
            
            # 决策头（保留用于其他用途，如果不需要可以删除）
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
            # 全局地图编码器 [B, 3, H, W] -> [B, 256]
            'map_backbone': nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
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
            
            # 单个智能体状态编码器 [4] -> [64]
            'agent_encoder': nn.Sequential(
                nn.Linear(4, 32),  # [x, y, x_g, y_g]
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            ),
            
            # 融合层：全局特征(256) + 智能体特征(64) -> 价值
            'fusion_layer': nn.Sequential(
                nn.Linear(256 + 64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        })
        
        # 每个智能体的独立价值输出头
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
    def critic_agent_encoder(self):
        return self.modules_dict['high_critic']['agent_encoder']
    
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
    def ego_node_encoder(self):
        return self.modules_dict['high_level']['ego_node_encoder']
    
    @property
    def explore_node_encoder(self):
        return self.modules_dict['high_level']['explore_node_encoder']
    
    @property
    def landmark_node_encoder(self):
        return self.modules_dict['high_level']['landmark_node_encoder']
    
    @property
    def edge_encoder(self):
        return self.modules_dict['high_level']['edge_encoder']
    
    @property
    def q_proj(self):
        return self.modules_dict['high_level']['q_proj']
    
    @property
    def k_proj(self):
        return self.modules_dict['high_level']['k_proj']
    
    @property
    def v_proj(self):
        return self.modules_dict['high_level']['v_proj']
    
    @property
    def node_selection_head(self):
        return self.modules_dict['high_level']['node_selection_head']
    
    @property
    def attn_dim(self):
        return 64

    def get_explore_nodes(self, vec_inp, map_inp, agent_indices=None, deterministic=False):
        """
        vec_inp: [Batch, num_agents, 4]，世界坐标：[x_pos, y_pos, x_goal, y_goal]
        map_inp: [2, Batch, num_agents, H, W], (0: entropy, 1: voronoi_mask)
        agent_indices: [N] 可选，仅在部分智能体更新时提供
        """

        # 1. 通过RRT生成候选目标点
        ## 输入vec_inp:[Batch, 2], map_inp:[2,Batch, H, W]
        ## 输出候选目标点B_candidate:[Batch, K, 3]，离散栅格坐标
        ### vec_inp转化为离散栅格坐标
        B_pro = vec_inp.size(0) 
        if agent_indices is not None:
            # agent_indices: [B_pro]
            batch_idx = torch.arange(B_pro, device=vec_inp.device)
            
            # [B_pro, num_agents, 4] -> [B_pro, 1, 4]
            update_nodes = vec_inp[batch_idx, agent_indices].unsqueeze(1)
            
            # [2, B_pro, num_agents, H, W] -> [B_pro, 1, H, W]
            voronoi_np = map_inp[1, batch_idx, agent_indices].unsqueeze(1).detach().cpu().numpy().astype(bool)
            entropy_np = map_inp[0, batch_idx, agent_indices].unsqueeze(1).detach().cpu().numpy().astype(np.float32)
        else:
            update_nodes = vec_inp
            voronoi_np = map_inp[1].detach().cpu().numpy().astype(bool)
            entropy_np = map_inp[0].detach().cpu().numpy().astype(np.float32)

        B_agents = update_nodes.size(1) # 如果指定了indices，这里是1；否则是num_agents        

        # update_nodes: [B_pro, B_agents, 4] -> 选取位置部分 [B_pro*B_agents, 2]
        starte_nodes = self._world_to_grid_torch(update_nodes.view(-1, update_nodes.size(2))[:, :2], H=100, W=100)  # [B_pro*B_agents, 2] 整数栅格坐标
        voronoi_inp = voronoi_np.reshape(-1, voronoi_np.shape[-2], voronoi_np.shape[-1])  # [B_pro*B_agents, H, W]
        entropy_inp = entropy_np.reshape(-1, entropy_np.shape[-2], entropy_np.shape[-1])  # [B_pro*B_agents, H, W]
        
        batch_rtt = plan_batch(starte_nodes, voronoi_inp, entropy_inp, max_iterations=60, top_k=5)  # [B_pro*B_agents, K, 3]
        batch_rtt = torch.tensor(batch_rtt, dtype=torch.long, device=vec_inp.device).view(B_pro, B_agents, -1, 3)  # [B_pro, B_agents, K, 3]
        
        ## 转为世界坐标
        explore_nodes_world = self._grid_to_world_torch(batch_rtt[..., :2].float(), H=100, W=100)  # [B_pro, B_agents, K, 2]
        
        # update_nodes: [B_pro, B_agents, 4]，前两维是 ego 的位置
        ego_positions = update_nodes[..., :2]  # [B_pro, B_agents, 2]
        
        # 广播减法：explore_nodes_world [B_pro, B_agents, K, 2] - ego_positions [B_pro, B_agents, 1, 2]
        relative_explore_positions = explore_nodes_world - ego_positions.unsqueeze(2)  # [B_pro, B_agents, K, 2]
        
        ## 拼接成候选点特征（现在前两维是相对位置）
        explore_nodes = torch.cat([relative_explore_positions, batch_rtt[..., 2:3].float()], dim=-1)  # [B_pro, B_agents, K, 3]

        ## 补充候选点节点特征 - Occupied Feature
        d0 = 0.3
        # 提取所有智能体的目标位置 [B_pro, num_agents, 2]
        all_goals = vec_inp[..., 2:4]        
        
        # 计算距离矩阵（这里仍然使用绝对位置计算距离）
        # explore_nodes_world: [B_pro, B_agents, K, 2]（绝对位置）
        # all_goals: [B_pro, num_agents, 2]（绝对位置）
        dists = torch.norm(explore_nodes_world.unsqueeze(3) - all_goals.unsqueeze(1).unsqueeze(1), dim=-1)

        mask = torch.ones_like(dists, dtype=torch.bool)
        if agent_indices is not None:
            batch_idx = torch.arange(B_pro, device=vec_inp.device)
            mask[batch_idx, 0, :, agent_indices] = False
        else:
            diag_mask = torch.eye(B_agents, device=vec_inp.device).bool()
            mask = ~diag_mask.view(1, B_agents, 1, B_agents).expand(B_pro, B_agents, 5, B_agents)

        dists = torch.where(mask, dists, torch.tensor(float('inf'), device=dists.device))

        valid_mask = (dists < d0)
        occ_vals = ((d0 - dists) / d0).pow(2)
        occ_vals = torch.where(valid_mask, occ_vals, torch.tensor(0.0, device=dists.device))
        
        occupied_feature = occ_vals.sum(dim=-1, keepdim=True) # [B_pro, B_agents, K, 1]
        
        # ✅ 最终的 explore_nodes: [B_pro, B_agents, K, 4]
        # 前两维是相对位置，第三维是熵值，第四维是 occupied 特征
        explore_nodes = torch.cat([explore_nodes, occupied_feature], dim=-1)  # [B_pro, B_agents, K, 4]

        return explore_nodes

    def get_landmark_nodes(self, agent_positions, detected, detected_mask, linear_indices):
        """
        agent_positions: [num_agents * num_processes, 2]，世界坐标
        detected: [num_agents * num_processes, max_landmarks, 4]，landmark特征 [x, y, utility, is_targeted]
        detected_mask: [num_agents * num_processes, max_landmarks, 1]，候选点有效掩码
        linear_indices: [N]，需要决策的智能体的线性索引

        return:
            batch_landmark_nodes: List of Tensors, 每个 Tensor 形状为 [L_i, 4]，一共Batch个需要决策智能体
                                  特征为 [relative_x, relative_y, utility, is_targeted]
            batch_landmark_nodes_masks: List of Tensors, 每个 Tensor 形状为 [L_i, 1]，一共Batch个需要决策智能体
            如果有智能体没有有效 landmark，则对应的 Tensor 形状为 [1, 4] 和 [1, 1]，内容全0（后续会被mask掉）
        """
        # 从更新后的 landmark 数据中提取对应的 landmarks
        batch_landmark_data = detected[linear_indices]  # [N, max_landmarks, 4]
        batch_landmark_mask = detected_mask[linear_indices]  # [N, max_landmarks, 1]
        
        # 提取对应智能体的位置 [N, 2]
        ego_positions = agent_positions[linear_indices]  # [N, 2]

        batch_landmark_nodes = []
        batch_landmark_nodes_masks = []

        for i in range(len(linear_indices)):
            # 获取有效的 landmarks
            valid_mask = batch_landmark_mask[i, :, 0] > 0.5  # [max_landmarks]
            valid_landmarks = batch_landmark_data[i, valid_mask]  # [L_i, 4]，L_i 是该智能体的有效 landmark 数量
            
            if valid_landmarks.shape[0] > 0:
                # 提取 landmark 的世界坐标 [L_i, 2]
                landmark_world_pos = valid_landmarks[:, :2]
                
                # 计算相对位置：landmark_pos - agent_pos
                # ego_positions[i]: [2]
                # landmark_world_pos: [L_i, 2]
                relative_pos = landmark_world_pos - ego_positions[i].unsqueeze(0)  # [L_i, 2]
                
                # 拼接相对位置和其他特征（utility, is_targeted）
                landmark_features = torch.cat([
                    relative_pos,              # [L_i, 2] - 相对位置
                    valid_landmarks[:, 2:]     # [L_i, 2] - utility, is_targeted
                ], dim=-1)  # [L_i, 4]
                
                batch_landmark_nodes.append(landmark_features)
                # 创建全1的mask，表示这些都是有效节点
                batch_landmark_nodes_masks.append(torch.ones(valid_landmarks.shape[0], 1, device=valid_landmarks.device))
            else:
                # 如果没有有效 landmark，添加一个占位符（后续会被mask掉）
                batch_landmark_nodes.append(torch.zeros(1, 4, device=batch_landmark_data.device))
                batch_landmark_nodes_masks.append(torch.zeros(1, 1, device=batch_landmark_data.device))

        return batch_landmark_nodes, batch_landmark_nodes_masks

    def get_edge_features(self, explore_nodes, landmark_nodes, landmark_node_masks, norm=False, max_distance=2.8):
        """
        计算边特征
        
        Args:
            explore_nodes: [B, K, 4]，候选探索点特征
            landmark_nodes: List of [L_i, 4]，landmark 特征，每个L不固定
            landmark_node_masks: List of [L_i, 1]，landmark 有效掩码
            norm: 是否归一化距离，默认False（不归一化）
            max_distance: 归一化时的最大距离，默认2.8
            
        Returns:
            batch_ego_to_explore_edges: List of [K, 3]
            batch_ego_to_landmark_edges: List of [L_i, 3]
            batch_ego_to_landmark_edge_masks: List of [L_i, 1]
        """
        # 1. ego -> explore nodes 的边特征
        batch_ego_to_explore_edges = []  # List of [K, 3]
        
        for i in range(explore_nodes.shape[0]):
            explore_relative_pos = explore_nodes[i, :, :2]  # [K, 2] - 相对位置
            
            # 计算距离
            distances = torch.norm(explore_relative_pos, dim=1, keepdim=True)  # [K, 1]
            
            # 根据 norm 选择距离特征
            if norm:
                d_feature = torch.clamp(distances / max_distance, max=1.0)  # [K, 1] 归一化
            else:
                d_feature = distances  # [K, 1] 原始距离
            
            # 计算角度的 cos 和 sin（避免除零）
            distances_safe = distances.clamp(min=1e-6)
            cos_theta = explore_relative_pos[:, 0:1] / distances_safe  # [K, 1]
            sin_theta = explore_relative_pos[:, 1:2] / distances_safe  # [K, 1]
            
            # 拼接边特征 [d, cos(θ), sin(θ)]
            edge_features = torch.cat([d_feature, cos_theta, sin_theta], dim=1)  # [K, 3]
            batch_ego_to_explore_edges.append(edge_features)
        
        # batch_ego_to_explore_edges: List of [K, 3], 长度为 N
        
        # 2. ego -> landmark nodes 的边特征
        batch_ego_to_landmark_edges = []  # List of [L_i, 3]
        batch_ego_to_landmark_edge_masks = []  # List of [L_i, 1]
        
        for i in range(len(landmark_nodes)):
            landmark_relative_pos = landmark_nodes[i][:, :2]  # [L_i, 2] - 相对位置
            landmark_mask = landmark_node_masks[i]  # [L_i, 1]
            
            # 计算距离
            distances = torch.norm(landmark_relative_pos, dim=1, keepdim=True)  # [L_i, 1]
            
            # 根据 norm 选择距离特征
            if norm:
                d_feature = torch.clamp(distances / max_distance, max=1.0)  # [L_i, 1] 归一化
            else:
                d_feature = distances  # [L_i, 1] 原始距离
            
            # 计算角度的 cos 和 sin（避免除零）
            distances_safe = distances.clamp(min=1e-6)
            cos_theta = landmark_relative_pos[:, 0:1] / distances_safe  # [L_i, 1]
            sin_theta = landmark_relative_pos[:, 1:2] / distances_safe  # [L_i, 1]
            
            # 拼接边特征 [d, cos(θ), sin(θ)]
            edge_features = torch.cat([d_feature, cos_theta, sin_theta], dim=1)  # [L_i, 3]
            
            batch_ego_to_landmark_edges.append(edge_features)
            batch_ego_to_landmark_edge_masks.append(landmark_mask)
        
        return batch_ego_to_explore_edges, batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks

    def get_high_level_goal(self, batch_ego_nodes, 
                            batch_explore_nodes, batch_ego_to_explore_edges, 
                            batch_landmark_nodes, batch_landmark_node_masks, 
                            batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks, 
                            deterministic=False):
        """
        统一从所有候选节点（explore + landmark）中选择一个目标
        
        Args:
            batch_ego_nodes: [B, 5] [x, y, vel_x, vel_y, battery]
            batch_explore_nodes: [B, K, 4] [relative_x, relative_y, utility, occupied]
            batch_ego_to_explore_edges: [B, K, 3] [d, cosθ, sinθ]
            batch_landmark_nodes: List[Tensor], 每个 [L_i, 4] [relative_x, relative_y, utility, is_targeted]
            batch_landmark_node_masks: List[Tensor], 每个 [L_i, 1] (1=有效, 0=无效)
            batch_ego_to_landmark_edges: List[Tensor], 每个 [L_i, 3]
            batch_ego_to_landmark_edge_masks: List[Tensor], 每个 [L_i, 1]
            deterministic: bool
        
        Returns:
            dict:
                action_modes: [B, 1] (0=explore, 1=landmark) 被选中节点的类型
                waypoints: [B, 2] 绝对世界坐标
                decision_log_probs: [B, 1] 节点选择的 log_prob
                map_log_probs: [B, 1] 同上（保持接口兼容）
        """
        B = batch_ego_nodes.size(0)
        K = batch_explore_nodes.size(1)
        
        # ===== 1. 节点和边的特征嵌入 =====
        # 1.1 Ego节点嵌入
        ego_node_feats = self.ego_node_encoder(batch_ego_nodes)  # [B, 64]
        
        # 1.2 Explore节点嵌入
        explore_node_feats = self.explore_node_encoder(
            batch_explore_nodes.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]
        
        explore_edges_tensor = torch.stack(batch_ego_to_explore_edges, dim=0) # [B, K, 3]
        explore_edge_feats = self.edge_encoder(
            explore_edges_tensor.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 32]
        
        # 1.3 Landmark节点嵌入（可变长度）
        landmark_node_feats_list = []  # 存储每个样本的 landmark 特征
        landmark_edge_feats_list = []
        landmark_lengths = []  # 记录每个样本的有效 landmark 数量
        
        for i in range(B):
            node_feat = self.landmark_node_encoder(batch_landmark_nodes[i])  # [L_i, 64]
            edge_feat = self.edge_encoder(batch_ego_to_landmark_edges[i])    # [L_i, 32]
            landmark_node_feats_list.append(node_feat)
            landmark_edge_feats_list.append(edge_feat)
            landmark_lengths.append(node_feat.size(0))
        
        max_L = max(landmark_lengths) if landmark_lengths else 0
        
        # ===== 2. 构建统一的候选节点集合 =====
        # 将 explore 和 landmark 合并为一个统一的节点集
        # 总节点数 = K (explore) + max_L (landmark)
        total_nodes = K + max_L
        
        # 2.1 准备 Query
        q = self.q_proj(ego_node_feats).unsqueeze(1)  # [B, 1, 64]
        
        # 2.2 合并所有节点的 Key 和 Value
        # 初始化统一的 K/V 矩阵: [B, K+max_L, 64]
        unified_k = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_v = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_mask = torch.zeros(B, total_nodes, device=ego_node_feats.device, dtype=torch.bool)
        
        # 存储相对坐标用于后续输出
        unified_relative_pos = torch.zeros(B, total_nodes, 2, device=ego_node_feats.device)
        
        # 节点类型标签: 0=explore, 1=landmark
        node_type_labels = torch.zeros(B, total_nodes, device=ego_node_feats.device, dtype=torch.long)
        
        # 填充 explore 节点 (索引 0 ~ K-1)
        explore_kv = torch.cat([explore_node_feats, explore_edge_feats], dim=-1)  # [B, K, 96]
        unified_k[:, :K, :] = self.k_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_v[:, :K, :] = self.v_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_mask[:, :K] = True  # explore 节点全部有效
        unified_relative_pos[:, :K, :] = batch_explore_nodes[:, :, :2]  # 相对坐标
        node_type_labels[:, :K] = 0  # explore 类型
        
        # 填充 landmark 节点 (索引 K ~ K+max_L-1)
        for i in range(B):
            Li = landmark_lengths[i]
            if Li > 0:
                lm_kv = torch.cat([landmark_node_feats_list[i], landmark_edge_feats_list[i]], dim=-1)  # [Li, 96]
                unified_k[i, K:K+Li, :] = self.k_proj(lm_kv)
                unified_v[i, K:K+Li, :] = self.v_proj(lm_kv)
                # 有效的 landmark：必须 mask 有效且未被追踪 (is_targeted=0)
                valid_mask = batch_landmark_node_masks[i][:, 0] > 0.5
                not_targeted = batch_landmark_nodes[i][:, 3] < 0.5  # is_targeted 在索引3，<0.5表示未被追踪
                combined_mask = valid_mask & not_targeted
                unified_mask[i, K:K+Li] = combined_mask
                unified_relative_pos[i, K:K+Li, :] = batch_landmark_nodes[i][:, :2]
                node_type_labels[i, K:K+Li] = 1  # landmark 类型
        
        # ===== 3. 注意力机制 =====
        # 计算注意力分数
        scale = math.sqrt(self.attn_dim)
        attn_scores = (q @ unified_k.transpose(1, 2)) / scale  # [B, 1, total_nodes]
        
        # 应用 mask（无效节点设为 -inf）
        attn_scores = attn_scores.masked_fill(~unified_mask.unsqueeze(1), -1e9)
        
        # Softmax 得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, total_nodes]
        
        # 注意力加权求和
        context = (attn_weights @ unified_v).squeeze(1)  # [B, 64]
        
        # ===== 4. 节点选择 =====
        # 对所有节点进行打分
        selection_logits = self.node_selection_head(unified_v).squeeze(-1)  # [B, total_nodes]
        
        # 应用 mask
        selection_logits = selection_logits.masked_fill(~unified_mask, -1e9)
        
        # 构建分类分布
        node_dist = TorchCategorical(logits=selection_logits)
        
        # 采样或选择最优节点
        if deterministic:
            selected_idx = torch.argmax(selection_logits, dim=-1)  # [B]
        else:
            selected_idx = node_dist.sample()  # [B] or [B, 1]
        
        # 确保 selected_idx 是 [B] 形状
        selected_idx = selected_idx.view(B)  # [B]
        
        # 计算 log_prob
        node_log_prob = node_dist.log_prob(selected_idx)  # [B]
        
        # ===== 5. 提取选中节点的信息 =====
        batch_indices = torch.arange(B, device=ego_node_feats.device)
        
        # 5.1 节点类型 (0=explore, 1=landmark)
        selected_type = node_type_labels[batch_indices, selected_idx]  # [B]
        
        # 5.2 相对坐标
        selected_relative_pos = unified_relative_pos[batch_indices, selected_idx, :]  # [B, 2]
        
        # 5.3 转换为绝对世界坐标
        ego_pos = batch_ego_nodes[:, :2]  # [B, 2]
        waypoints_world = ego_pos + selected_relative_pos  # [B, 2]
        
        # ===== 6. 返回结果 =====
        return {
            "action_modes": selected_type.unsqueeze(-1),       # [B, 1] 节点类型
            "waypoints": waypoints_world,                      # [B, 2] 绝对世界坐标
            "node_log_probs": node_log_prob.unsqueeze(-1), # [B, 1] 节点选择 log_prob
        }
    
    def get_high_value(self, map_inp, agent_states):
        """
        计算每个智能体的状态价值
        
        Args:
            map_inp: [B, 3, H, W] 全局地图 (entropy_map, heatmap, landmark_heatmap)
            agent_states: [B, num_agents, 4] 智能体状态 [x, y, x_g, y_g]
        
        Returns:
            values: [B, num_agents] 每个智能体的价值估计
        """
        B = map_inp.size(0)
        num_agents = agent_states.size(1)

        # 1. 全局地图特征提取 [B, 3, H, W] -> [B, 256]
        f_map = self.critic_map_backbone(map_inp)  # [B, 64, 6, 6]
        f_map_flat = f_map.view(B, -1)  # [B, 64*6*6]
        f_global = self.critic_map_compress(f_map_flat)  # [B, 256]

        # 2. 为每个智能体计算价值
        values = []
        for agent_idx in range(num_agents):
            # 2.1 提取该智能体的状态 [B, 4]
            agent_state = agent_states[:, agent_idx, :]  # [B, 4]
            
            # 2.2 编码智能体状态 [B, 4] -> [B, 64]
            f_agent = self.critic_agent_encoder(agent_state)  # [B, 64]
            
            # 2.3 融合全局特征和智能体特征 [B, 256] + [B, 64] -> [B, 320]
            fused = torch.cat([f_global, f_agent], dim=1)  # [B, 320]
            
            # 2.4 通过融合层 [B, 320] -> [B, 128]
            h = self.critic_fusion_layer(fused)  # [B, 128]
            
            # 2.5 通过该智能体的独立价值头 [B, 128] -> [B, 1]
            value = self.critic_value_out_heads[agent_idx](h)  # [B, 1]
            values.append(value)
        
        # 3. 拼接所有智能体的价值 [B, num_agents]
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

    def evaluate_high_actions(self, env_states, obs,
                              critic_maps, critic_nodes, goals, tasks, 
                              ego_nodes, explore_nodes, landmark_datas, landmark_masks, 
                              agent_ids):
        """
        评估给定高层动作的log_prob、熵和价值（用于PPO更新）
        
        Input:
            env_states: [batch, env_dim]
            obs: [batch, obs_dim] - 用于提取智能体位置
            critic_maps: [batch, 3, H, W] - 用于critic
            critic_nodes: [batch, num_agents, 4] - 用于critic
            goals: [batch, 2] - 已选择的目标位置（世界坐标）
            tasks: [batch, 1] - 已选择的任务类型（0=explore, 1=landmark）
            ego_nodes: [batch, 5]
            explore_nodes: [batch, K, 4]
            landmark_datas: [batch, num_landmarks, 4]
            landmark_masks: [batch, num_landmarks, 1]
            agent_ids: [batch, 1] - 智能体ID
            
        Returns:
            high_values: [batch, 1] - 状态价值
            node_log_probs: [batch, 1] - 给定节点选择的log概率
            node_entropy: [batch, 1] - 节点选择分布的熵
        """
        batch_size = env_states.size(0)
        num_agents = self.num_agents
        K = explore_nodes.size(1)  # explore节点数量

        # =====================================================
        # 1. 重建Graph nodes和Edges
        # =====================================================
        # 获取 landmark nodes 和 masks
        landmark_nodes, landmark_node_masks = self.get_landmark_nodes(
            agent_positions=obs[:, 2:4],  # 提取智能体当前位置
            detected=landmark_datas,    
            detected_mask=landmark_masks,
            linear_indices=torch.arange(batch_size, device=env_states.device)
        )

        # 计算边特征
        ego_to_explore_edges, ego_to_landmark_edges, ego_to_landmark_edge_masks = self.get_edge_features(
            explore_nodes=explore_nodes,
            landmark_nodes=landmark_nodes,
            landmark_node_masks=landmark_node_masks,
            norm=False,
            max_distance=2.8
        )

        # =====================================================
        # 2. 复用 get_high_level_goal 的逻辑构建节点分布
        # =====================================================
        B = batch_size
        
        # 2.1 节点和边的特征嵌入
        ego_node_feats = self.ego_node_encoder(ego_nodes)  # [B, 64]
        
        explore_node_feats = self.explore_node_encoder(
            explore_nodes.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]
        
        explore_edges_tensor = torch.stack(ego_to_explore_edges, dim=0)
        explore_edge_feats = self.edge_encoder(
            explore_edges_tensor.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 32]
        
        landmark_node_feats_list = []
        landmark_edge_feats_list = []
        landmark_lengths = []
        
        for i in range(B):
            node_feat = self.landmark_node_encoder(landmark_nodes[i])
            edge_feat = self.edge_encoder(ego_to_landmark_edges[i])
            landmark_node_feats_list.append(node_feat)
            landmark_edge_feats_list.append(edge_feat)
            landmark_lengths.append(node_feat.size(0))
        
        max_L = max(landmark_lengths) if landmark_lengths else 0
        
        # 2.2 构建统一的候选节点集合
        total_nodes = K + max_L
        
        q = self.q_proj(ego_node_feats).unsqueeze(1)  # [B, 1, 64]
        
        unified_k = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_v = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_mask = torch.zeros(B, total_nodes, device=ego_node_feats.device, dtype=torch.bool)
        unified_relative_pos = torch.zeros(B, total_nodes, 2, device=ego_node_feats.device)
        
        # 填充 explore 节点
        explore_kv = torch.cat([explore_node_feats, explore_edge_feats], dim=-1)
        unified_k[:, :K, :] = self.k_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_v[:, :K, :] = self.v_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_mask[:, :K] = True
        unified_relative_pos[:, :K, :] = explore_nodes[:, :, :2]
        
        # 填充 landmark 节点
        for i in range(B):
            Li = landmark_lengths[i]
            if Li > 0:
                lm_kv = torch.cat([landmark_node_feats_list[i], landmark_edge_feats_list[i]], dim=-1)
                unified_k[i, K:K+Li, :] = self.k_proj(lm_kv)
                unified_v[i, K:K+Li, :] = self.v_proj(lm_kv)
                valid_mask = landmark_node_masks[i][:, 0] > 0.5
                not_targeted = landmark_nodes[i][:, 3] < 0.5
                combined_mask = valid_mask & not_targeted
                unified_mask[i, K:K+Li] = combined_mask
                unified_relative_pos[i, K:K+Li, :] = landmark_nodes[i][:, :2]
        
        # 2.3 注意力机制
        scale = math.sqrt(self.attn_dim)
        attn_scores = (q @ unified_k.transpose(1, 2)) / scale
        attn_scores = attn_scores.masked_fill(~unified_mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ unified_v).squeeze(1)
        
        # 2.4 节点选择分布
        selection_logits = self.node_selection_head(unified_v).squeeze(-1)  # [B, total_nodes]
        selection_logits = selection_logits.masked_fill(~unified_mask, -1e9)
        node_dist = TorchCategorical(logits=selection_logits)
        
        # =====================================================
        # 3. 找到给定 goal 对应的节点索引
        # =====================================================
        # 将给定的 goals (世界坐标) 转换为相对坐标
        ego_pos = ego_nodes[:, :2]  # [B, 2]
        goals_relative = goals - ego_pos  # [B, 2]
        
        # 计算 goals_relative 到所有节点的距离
        # unified_relative_pos: [B, total_nodes, 2]
        # goals_relative: [B, 2] -> [B, 1, 2]
        dists = torch.norm(unified_relative_pos - goals_relative.unsqueeze(1), dim=-1)  # [B, total_nodes]
        
        # 只在有效节点中查找最近的
        dists_masked = dists.masked_fill(~unified_mask, float('inf'))
        selected_idx = torch.argmin(dists_masked, dim=-1)  # [B]
        
        # =====================================================
        # 4. 计算 log_prob 和 entropy
        # =====================================================
        node_log_probs = node_dist.log_prob(selected_idx).unsqueeze(-1)  # [B, 1]
        node_entropy = node_dist.entropy().unsqueeze(-1)  # [B, 1]
        
        # =====================================================
        # 5. 计算 Critic 价值
        # =====================================================
        # 计算所有智能体的价值 [B, num_agents]
        all_values = self.get_high_value(critic_maps, critic_nodes)  # [B, num_agents]
        
        # 根据 agent_ids 选择对应的价值
        agent_ids_flat = agent_ids.squeeze(-1)  # [B]
        batch_indices = torch.arange(B, device=env_states.device)
        high_values = all_values[batch_indices, agent_ids_flat].unsqueeze(-1)  # [B, 1]

        return (high_values, node_log_probs, node_entropy)
    
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

    def _world_to_grid_torch(self, world_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        world_xy: [..., 2] in continuous world coords (x,y), assumed in [-arena_size/2, arena_size/2]
        return:  [..., 2] grid indices (i,j) as int64, clipped to [0..H-1/W-1]
        """
        world_min = -1.0
        cell_size_x = 2.0 / float(H)
        cell_size_y = 2.0 / float(W)

        x = world_xy[..., 0]
        y = world_xy[..., 1]

        i = torch.floor((x - world_min) / cell_size_x)
        j = torch.floor((y - world_min) / cell_size_y)

        i = i.clamp(0, H - 1)
        j = j.clamp(0, W - 1)

        return torch.stack([i, j], dim=-1).long()
    
    def _grid_to_world_torch(self, grid_ij: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        grid_ij: [..., 2] grid indices (i,j) as int64
        return: [..., 2] in continuous world coords (x,y), in [-arena_size/2, arena_size/2]
        """
        world_min = -1.0
        cell_size_x = 2.0 / float(H)
        cell_size_y = 2.0 / float(W)

        i = grid_ij[..., 0].float()
        j = grid_ij[..., 1].float()

        x = world_min + (i + 0.5) * cell_size_x
        y = world_min + (j + 0.5) * cell_size_y

        return torch.stack([x, y], dim=-1)  

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
