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
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=3, mask_dist=None, mask_obs_dist=None,
                 entity_mp=False, is_recurrent=True, high_actor_backbone='graph', high_mlp_hidden=256,
                 high_mlp_layers=3, high_mlp_dropout=0.0):
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
        self.high_actor_backbone = str(high_actor_backbone).lower()
        if self.high_actor_backbone not in ("graph", "mlp"):
            raise ValueError(f"Unsupported high_actor_backbone: {high_actor_backbone}")
        self.high_mlp_hidden = int(high_mlp_hidden)
        self.high_mlp_layers = int(high_mlp_layers)
        self.high_mlp_dropout = float(high_mlp_dropout)
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

    def _build_mlp(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0):
        """Build a configurable feed-forward MLP."""
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _build_high_level_modules(self):
        """构建高层策略网络（Actor）"""
        mlp_context_input_dim = 5 + self.num_agents * 6
        mlp_slot_input_dim = 9
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

            # teammate节点编码器 [5] -> [64]
            'teammate_node_encoder': nn.Sequential(
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

            # explore线性映射层 [64] -> [64]
            'explore_node_linear': nn.Linear(64, 64),

            # landmark线性映射层 [64] -> [64]
            'landmark_node_linear': nn.Linear(64, 64),

            'linear_ln': nn.LayerNorm(64),
            
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

            # ===== MLP 消融模块 =====
            # context: ego + teammate(+mask)
            'mlp_context_encoder': self._build_mlp(
                in_dim=mlp_context_input_dim,
                hidden_dim=self.high_mlp_hidden,
                out_dim=64,
                num_layers=self.high_mlp_layers,
                dropout=self.high_mlp_dropout
            ),
            # slot: [dx, dy, utility, flag, d, cos, sin, type, valid]
            'mlp_slot_encoder': self._build_mlp(
                in_dim=mlp_slot_input_dim,
                hidden_dim=self.high_mlp_hidden,
                out_dim=64,
                num_layers=self.high_mlp_layers,
                dropout=self.high_mlp_dropout
            ),
            # score each slot conditioned on context
            'mlp_selection_head': self._build_mlp(
                in_dim=128,
                hidden_dim=self.high_mlp_hidden,
                out_dim=1,
                num_layers=self.high_mlp_layers,
                dropout=self.high_mlp_dropout
            ),
            
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

    def load_pretrained_high_level(self, path, freeze=False):
        """
        加载预训练的高层策略（Actor + Critic）
        """
        print(f"🔄 Loading high-level params from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        high_level_state_dict = {}
        high_critic_state_dict = {}
        
        # === 情况 A: 模块化 Checkpoint ===
        if 'state_dict' in checkpoint:
            module_name = checkpoint.get('module_name', None)
            if module_name == 'high_level':
                high_level_state_dict = checkpoint['state_dict']
            elif module_name == 'high_critic':
                high_critic_state_dict = checkpoint['state_dict']

        # === 情况 B: 完整 Checkpoint ===
        elif 'models' in checkpoint:
            print("  Type: Full Training Checkpoint (extracting high-level params...)")
            full_state_dict = checkpoint['models'][0]
            
            for key, value in full_state_dict.items():
                if 'modules_dict.high_level.' in key:
                    new_key = key.replace('modules_dict.high_level.', '')
                    high_level_state_dict[new_key] = value
                elif 'modules_dict.high_critic.' in key:
                    new_key = key.replace('modules_dict.high_critic.', '')
                    high_critic_state_dict[new_key] = value
        
        # 执行加载
        if high_level_state_dict:
            m, u = self.modules_dict['high_level'].load_state_dict(high_level_state_dict, strict=False)
            print(f"  ✅ High-Level Actor loaded. Missing: {len(m)}, Unexpected: {len(u)}")
            if freeze: self.freeze_module('high_level')
            
        if high_critic_state_dict:
            m, u = self.modules_dict['high_critic'].load_state_dict(high_critic_state_dict, strict=False)
            print(f"  ✅ High-Level Critic loaded. Missing: {len(m)}, Unexpected: {len(u)}")
            if freeze: self.freeze_module('high_critic')

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
    def teammate_node_encoder(self):
        return self.modules_dict['high_level']['teammate_node_encoder']
    
    @property
    def explore_node_encoder(self):
        return self.modules_dict['high_level']['explore_node_encoder']
    
    @property
    def landmark_node_encoder(self):
        return self.modules_dict['high_level']['landmark_node_encoder']
    
    @property
    def explore_node_linear(self):
        return self.modules_dict['high_level']['explore_node_linear']
    
    @property
    def landmark_node_linear(self):
        return self.modules_dict['high_level']['landmark_node_linear']
    
    @property
    def linear_ln(self):
        return self.modules_dict['high_level']['linear_ln']
    
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
    def mlp_context_encoder(self):
        return self.modules_dict['high_level']['mlp_context_encoder']

    @property
    def mlp_slot_encoder(self):
        return self.modules_dict['high_level']['mlp_slot_encoder']

    @property
    def mlp_selection_head(self):
        return self.modules_dict['high_level']['mlp_selection_head']
    
    @property
    def attn_dim(self):
        return 64

    def get_explore_nodes(self, top_k, rrt_max_iter, vec_inp, map_inp, agent_indices=None, deterministic=False):
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
        
        batch_rtt = plan_batch(starte_nodes, voronoi_inp, entropy_inp, max_iterations=rrt_max_iter, top_k=top_k)  # [B_pro*B_agents, K, 3]
        sanitized_batch_rtt = []
        fallback_count = 0
        for start_node, sample in zip(starte_nodes.tolist(), batch_rtt):
            normalized_sample = []
            for node in sample[:top_k]:
                if len(node) < 3:
                    continue
                normalized_sample.append([float(node[0]), float(node[1]), float(node[2])])

            while len(normalized_sample) < top_k:
                fallback_count += 1
                fallback_x = float(np.clip(start_node[0] + np.random.randint(-5, 6), 0, voronoi_inp.shape[-2] - 1))
                fallback_y = float(np.clip(start_node[1] + np.random.randint(-5, 6), 0, voronoi_inp.shape[-1] - 1))
                normalized_sample.append([fallback_x, fallback_y, 0.0])

            sanitized_batch_rtt.append(normalized_sample)

        if fallback_count > 0:
            print(f"Warning: padded {fallback_count} RRT candidate slots to keep fixed top_k={top_k}")

        batch_rtt = torch.tensor(sanitized_batch_rtt, dtype=torch.float32, device=vec_inp.device).view(B_pro, B_agents, top_k, 3)  # [B_pro, B_agents, K, 3]
        
        ## 转为世界坐标
        explore_nodes_world = self._grid_to_world_torch(batch_rtt[..., :2], H=100, W=100)  # [B_pro, B_agents, K, 2]
        
        # update_nodes: [B_pro, B_agents, 4]，前两维是 ego 的位置
        ego_positions = update_nodes[..., :2]  # [B_pro, B_agents, 2]
        
        # 广播减法：explore_nodes_world [B_pro, B_agents, K, 2] - ego_positions [B_pro, B_agents, 1, 2]
        relative_explore_positions = explore_nodes_world - ego_positions.unsqueeze(2)  # [B_pro, B_agents, K, 2]
        
        ## 拼接成候选点特征（现在前两维是相对位置）
        explore_nodes = torch.cat([relative_explore_positions, batch_rtt[..., 2:3]], dim=-1)  # [B_pro, B_agents, K, 3]

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
            mask = ~diag_mask.view(1, B_agents, 1, B_agents).expand(B_pro, B_agents, top_k, B_agents)

        dists = torch.where(mask, dists, torch.tensor(float('inf'), device=dists.device))

        valid_mask = (dists < d0)
        occ_vals = ((d0 - dists) / d0).pow(2)
        occ_vals = torch.where(valid_mask, occ_vals, torch.tensor(0.0, device=dists.device))
        
        occupied_feature = occ_vals.sum(dim=-1, keepdim=True) # [B_pro, B_agents, K, 1]
        
        # ✅ 最终的 explore_nodes: [B_pro, B_agents, K, 4]
        # 前两维是相对位置，第三维是熵值，第四维是 occupied 特征
        explore_nodes = torch.cat([explore_nodes, occupied_feature], dim=-1)  # [B_pro, B_agents, K, 4]

        return explore_nodes

    def get_landmark_nodes(self, agent_positions, detected, detected_mask, linear_indices, all_masks=None):
        """
        agent_positions: [num_agents * num_processes, 2]，世界坐标
        detected: [num_agents * num_processes, max_landmarks, 4]，landmark特征 [x, y, utility, is_targeted]
        detected_mask: [num_agents * num_processes, max_landmarks, 1]，候选点有效掩码
        linear_indices: [N]，需要决策的智能体的线性索引
        all_masks: [num_agents * num_processes, 1] 可选，智能体有效掩码

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

        # ✅ 批量计算相对位置
        relative_pos = batch_landmark_data[:, :, :2] - ego_positions.unsqueeze(1)  # [N, max_L, 2]

        # 根据其他智能体是否已经被发现，判定当前的landmark效用值
        
        # 更新landmark数据（保留utility和is_targeted）
        batch_landmark_data_relative = torch.cat([
            relative_pos,
            batch_landmark_data[:, :, 2:]  # [N, max_L, 2] - utility和is_targeted
        ], dim=2)  # [N, max_L, 4]
        
        # ===== 动态调整 landmark 效用值（基于同环境中的退休智能体数量）=====
        if all_masks is not None:
            # 1. 计算每个智能体所在的环境索引
            # linear_indices: [N]，范围 [0, num_agents * num_processes)
            # 数据组织方式：[agent0_env0, agent0_env1, ..., agent0_env(P-1), agent1_env0, ...]
            # 环境索引 = linear_indices % num_processes
            num_processes = all_masks.size(0) // self.num_agents
            env_indices = linear_indices % num_processes  # [N]
            
            # 2. 为每个样本计算其所在环境中的退休智能体数量
            # all_masks: [num_agents * num_processes, 1]
            # 重塑为 [num_agents, num_processes] 然后转置为 [num_processes, num_agents]
            all_masks_reshaped = all_masks.view(self.num_agents, num_processes).t()  # [P, A]
            
            # 对于每个环境，统计无效（已退休）智能体数量
            # mask=1 表示有效，mask=0 表示已退休
            retired_counts_per_env = (all_masks_reshaped < 0.5).sum(dim=1)  # [P]
            
            # 3. 根据环境索引获取对应的退休智能体数量
            num_retired_agents = retired_counts_per_env[env_indices]  # [N]
            
            # 4. 提取 is_targeted 和 landmark 有效性掩码
            is_targeted = batch_landmark_data_relative[:, :, 3]  # [N, max_L]
            valid_landmark_mask = batch_landmark_mask.squeeze(-1) > 0.5  # [N, max_L]
            
            # 5. 对于有效且尚未被追踪的 landmark，设置效用值为该环境中的退休智能体数量
            untargeted_and_valid = (is_targeted < 0.5) & valid_landmark_mask  # [N, max_L]
            
            # 6. 更新效用值（广播 num_retired_agents [N] 到 [N, max_L]）
            vals = (num_retired_agents + 2).float().unsqueeze(1)  # [N, 1]
            batch_landmark_data_relative[:, :, 2] = torch.where(
                untargeted_and_valid,
                vals,
                batch_landmark_data_relative[:, :, 2]
            )
        
        # ===== 效用值调整完成 =====
        
        # ✅ 直接返回tensor，而不是list，[N, max_L, 4] 和 [N, max_L, 1]
        return batch_landmark_data_relative, batch_landmark_mask

    def get_edge_features(self, explore_nodes, landmark_nodes, landmark_node_masks, norm=False, max_distance=2.8):
        """
        计算边特征
        
        Args:
            explore_nodes: [B, K, 4]，候选探索点特征
            landmark_nodes: tensor: [B, Max_L, 4]，landmark 特征
            landmark_node_masks: tensor: [B, Max_L, 1]，landmark 有效掩码
            norm: 是否归一化距离，默认False（不归一化）
            max_distance: 归一化时的最大距离，默认2.8
            
        Returns:
            batch_ego_to_explore_edges: tensor: [N, K, 3]
            batch_ego_to_landmark_edges: tensor: [N, Max_L, 3]
            batch_ego_to_landmark_edge_masks: tensor: [N, Max_L, 1]
        """
        """向量化版本 - 批量处理所有样本"""
        B = explore_nodes.shape[0]
        K = explore_nodes.shape[1]

        # ✅ 批量计算explore边特征
        explore_relative_pos = explore_nodes[:, :, :2]  # [B, K, 2]
        distances = torch.norm(explore_relative_pos, dim=2, keepdim=True)  # [B, K, 1]

        if norm:
            d_feature = distances / max_distance
        else:
            d_feature = distances
        
        distances_safe = distances.clamp(min=1e-6)
        cos_theta = explore_relative_pos[:, :, 0:1] / distances_safe  # [B, K, 1]
        sin_theta = explore_relative_pos[:, :, 1:2] / distances_safe  # [B, K, 1]

        # [B, K, 3] - 直接返回tensor而不是list
        batch_ego_to_explore_edges = torch.cat([d_feature, cos_theta, sin_theta], dim=2)

        # ✅ 批量处理landmark边特征（使用padding）
        max_L = landmark_nodes.shape[1]
        
        # 构建padded tensor
        landmark_relative_pos = landmark_nodes[:, :, :2]  # [B, Max_L, 2]
        distances_landmark = torch.norm(landmark_relative_pos, dim=2, keepdim=True)  # [B, Max_L, 1]

        if norm:
            d_feature_landmark = distances_landmark / max_distance
        else:
            d_feature_landmark = distances_landmark 

        distances_landmark_safe = distances_landmark.clamp(min=1e-6)
        cos_theta_landmark = landmark_relative_pos[:, :, 0:1] / distances_landmark_safe  # [B, Max_L, 1]
        sin_theta_landmark = landmark_relative_pos[:, :, 1:2] / distances_landmark_safe  # [B, Max_L, 1]    

        batch_ego_to_landmark_edges = torch.cat([d_feature_landmark, cos_theta_landmark, sin_theta_landmark], dim=2)  # [B, Max_L, 3]
        batch_ego_to_landmark_edge_masks = landmark_node_masks  # [B, Max_L, 1]
        
        return batch_ego_to_explore_edges, batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
    
    def landmark_team_gat(self, batch_ego_pos, batch_teammate_nodes, batch_teammate_masks,
                           batch_landmark_nodes, batch_landmark_node_masks,
                           landmark_node_feats):
        """
        使用队友节点对 landmark 节点进行 GAT 聚合
        Args:
            batch_ego_pos: [B, 2] ego 位置，用于计算相对位置
            batch_teammate_nodes: [B, num_agents, 5] 队友节点特征,[x, y, vel_x, vel_y, dist_to_goal]
            batch_teammate_masks: [B, num_agents, 1] 队友节点掩码
            batch_landmark_nodes: [B, Max_L, 4] landmark 节点特征 (相对ego的位置)
            batch_landmark_node_masks: [B, Max_L, 1] landmark 节点掩码
            landmark_node_feats: [B, Max_L, 64] landmark 节点嵌入特征
        Returns:
            landmark_node_agg_feats: [B, Max_L, 64] 聚合后的 landmark 节点特征
        """
        B = batch_landmark_nodes.size(0)
        max_L = batch_landmark_nodes.size(1)
        num_agents = batch_teammate_nodes.size(1)

        # 1. 计算landmark的绝对世界坐标
        # batch_landmark_nodes[:, :, :2] 是相对于ego的位置 [B, Max_L, 2]
        # batch_ego_pos [B, 2] -> [B, 1, 2]
        landmark_abs_pos = batch_ego_pos.unsqueeze(1) + batch_landmark_nodes[:, :, :2]  # [B, Max_L, 2]
        
        # 2. 提取队友的绝对位置
        teammate_abs_pos = batch_teammate_nodes[:, :, :2]  # [B, num_agents, 2]
        
        # 3. 计算队友相对于每个landmark的位置
        # teammate_abs_pos [B, 1, num_agents, 2] - landmark_abs_pos [B, Max_L, 1, 2]
        # => [B, Max_L, num_agents, 2]
        teammate_relative_to_landmark = teammate_abs_pos.unsqueeze(1) - landmark_abs_pos.unsqueeze(2)  # [B, Max_L, num_agents, 2]
        
        # 4. 拼接其他特征（速度和距离目标）
        # batch_teammate_nodes[:, :, 2:] [B, num_agents, 3] -> [B, 1, num_agents, 3]
        teammate_other_feats = batch_teammate_nodes[:, :, 2:].unsqueeze(1).expand(B, max_L, num_agents, 3)  # [B, Max_L, num_agents, 3]
        
        # 5. 组合成完整的队友节点特征 [B, Max_L, num_agents, 5]
        teammate_nodes_per_landmark = torch.cat([
            teammate_relative_to_landmark,  # [B, Max_L, num_agents, 2]
            teammate_other_feats            # [B, Max_L, num_agents, 3]
        ], dim=-1)  # [B, Max_L, num_agents, 5]

        # 6. 编码队友节点特征
        teammate_node_feats = self.teammate_node_encoder(
            teammate_nodes_per_landmark.contiguous().view(B * max_L * num_agents, -1)
        ).view(B, max_L, num_agents, -1)  # [B, Max_L, num_agents, 64]

        # 7. 通过多头注意力机制对landmark节点进行聚合
        # landmark节点作为Query，队友节点作为Key和Value
        
        # 7.1 准备Query：landmark节点特征 [B, Max_L, 64] -> [B, Max_L, 1, 64]
        landmark_query = landmark_node_feats.unsqueeze(2)  # [B, Max_L, 1, 64]
        
        # 7.2 准备Key和Value：队友节点特征 [B, Max_L, num_agents, 64]
        teammate_key = teammate_node_feats  # [B, Max_L, num_agents, 64]
        teammate_value = teammate_node_feats  # [B, Max_L, num_agents, 64]
        
        # 7.3 计算注意力分数
        scale = math.sqrt(64)  # attn_dim = 64
        attn_scores = torch.matmul(landmark_query, teammate_key.transpose(-2, -1)) / scale  # [B, Max_L, 1, num_agents]
        
        # 7.4 应用队友节点mask（将无效队友的注意力分数设为-inf）
        # batch_teammate_masks: [B, num_agents, 1] -> [B, 1, 1, num_agents]
        teammate_mask_expanded = batch_teammate_masks.transpose(1, 2).unsqueeze(1)  # [B, 1, 1, num_agents]
        teammate_mask_expanded = teammate_mask_expanded.expand(B, max_L, 1, num_agents)  # [B, Max_L, 1, num_agents]
        attn_scores = attn_scores.masked_fill(teammate_mask_expanded < 0.5, -1e9)
        
        # 7.5 应用landmark节点mask（无效的landmark不计算注意力）
        # batch_landmark_node_masks: [B, Max_L, 1] -> [B, Max_L, 1, 1]
        landmark_mask_expanded = batch_landmark_node_masks.unsqueeze(-1)  # [B, Max_L, 1, 1]
        attn_scores = attn_scores.masked_fill(landmark_mask_expanded < 0.5, -1e9)
        
        # 7.6 Softmax得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, Max_L, 1, num_agents]

        # 更严谨的做法：重新应用 Mask 到 weights 上，确保被 mask 的权重绝对为 0
        attn_weights = attn_weights * teammate_mask_expanded
        
        # 7.7 加权求和得到聚合特征
        landmark_aggregated = torch.matmul(attn_weights, teammate_value).squeeze(2)  # [B, Max_L, 64]

        # 构造一个聚合 Mask：如果在该维度上所有 teammate 都被 mask，则该 aggregate 无效
        # teammate_mask_expanded: [B, Max_L, 1, num_agents]
        # valid_context_mask: [B, Max_L, 1] -> True 表示至少有一个队友
        valid_context_mask = (teammate_mask_expanded.sum(dim=-1) > 0).float() # [B, Max_L, 1]
        
        # 强制清零无效的聚合
        landmark_aggregated = landmark_aggregated * valid_context_mask
        
        # 7.8 残差连接：聚合特征 + 原始landmark特征
        landmark_node_agg_feats = landmark_node_feats + landmark_aggregated  # [B, Max_L, 64]
        
        return landmark_node_agg_feats

    def _sort_high_level_nodes_for_mlp(self,
                                       batch_explore_nodes, batch_ego_to_explore_edges,
                                       batch_landmark_nodes, batch_landmark_node_masks,
                                       batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks):
        """Sort explore by utility(desc) and landmark by distance(asc) for MLP slots."""
        B, K = batch_explore_nodes.size(0), batch_explore_nodes.size(1)
        max_L = batch_landmark_nodes.size(1)

        if K > 1:
            explore_sort_idx = torch.argsort(batch_explore_nodes[:, :, 2], dim=1, descending=True)
            gather_idx = explore_sort_idx.unsqueeze(-1)
            batch_explore_nodes = torch.gather(
                batch_explore_nodes, 1, gather_idx.expand(B, K, batch_explore_nodes.size(-1))
            )
            batch_ego_to_explore_edges = torch.gather(
                batch_ego_to_explore_edges, 1, gather_idx.expand(B, K, batch_ego_to_explore_edges.size(-1))
            )

        if max_L > 1:
            valid_landmark_mask = batch_landmark_node_masks[:, :, 0] > 0.5
            landmark_dist = batch_ego_to_landmark_edges[:, :, 0].masked_fill(~valid_landmark_mask, float('inf'))
            landmark_sort_idx = torch.argsort(landmark_dist, dim=1, descending=False)
            gather_idx = landmark_sort_idx.unsqueeze(-1)
            batch_landmark_nodes = torch.gather(
                batch_landmark_nodes, 1, gather_idx.expand(B, max_L, batch_landmark_nodes.size(-1))
            )
            batch_landmark_node_masks = torch.gather(
                batch_landmark_node_masks, 1, gather_idx.expand(B, max_L, batch_landmark_node_masks.size(-1))
            )
            batch_ego_to_landmark_edges = torch.gather(
                batch_ego_to_landmark_edges, 1, gather_idx.expand(B, max_L, batch_ego_to_landmark_edges.size(-1))
            )
            batch_ego_to_landmark_edge_masks = torch.gather(
                batch_ego_to_landmark_edge_masks, 1, gather_idx.expand(B, max_L, batch_ego_to_landmark_edge_masks.size(-1))
            )

        return (
            batch_explore_nodes, batch_ego_to_explore_edges,
            batch_landmark_nodes, batch_landmark_node_masks,
            batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
        )

    def _build_unified_slot_tensors(self,
                                    batch_explore_nodes, batch_ego_to_explore_edges,
                                    batch_landmark_nodes, batch_landmark_node_masks,
                                    batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks):
        """Build fixed-length candidate slots shared by graph/mlp high-level actors."""
        B = batch_explore_nodes.size(0)
        K = batch_explore_nodes.size(1)
        max_L = batch_landmark_nodes.size(1)

        explore_valid = torch.ones(B, K, device=batch_explore_nodes.device, dtype=torch.bool)
        valid_mask = batch_landmark_node_masks[:, :, 0] > 0.5
        edge_valid_mask = batch_ego_to_landmark_edge_masks[:, :, 0] > 0.5
        not_targeted = batch_landmark_nodes[:, :, 3] < 0.5
        landmark_valid = valid_mask & edge_valid_mask & not_targeted

        # Keep original behavior: if all landmarks are already found, disable explore nodes.
        all_landmarks_found = valid_mask.all(dim=1)
        if all_landmarks_found.any():
            explore_valid[all_landmarks_found, :] = False

        unified_mask = torch.cat([explore_valid, landmark_valid], dim=1)  # [B, K+max_L]
        unified_relative_pos = torch.cat(
            [batch_explore_nodes[:, :, :2], batch_landmark_nodes[:, :, :2]], dim=1
        )  # [B, K+max_L, 2]

        node_type_labels = torch.cat([
            torch.zeros(B, K, device=batch_explore_nodes.device, dtype=torch.long),
            torch.ones(B, max_L, device=batch_explore_nodes.device, dtype=torch.long)
        ], dim=1)  # [B, K+max_L]

        explore_type = torch.zeros(B, K, 1, device=batch_explore_nodes.device)
        landmark_type = torch.ones(B, max_L, 1, device=batch_explore_nodes.device)
        explore_valid_f = explore_valid.float().unsqueeze(-1)
        landmark_valid_f = landmark_valid.float().unsqueeze(-1)

        # slot feats: [dx, dy, utility, flag, d, cos, sin, type, valid]
        explore_slots = torch.cat(
            [batch_explore_nodes, batch_ego_to_explore_edges, explore_type, explore_valid_f], dim=-1
        )  # [B, K, 9]
        landmark_slots = torch.cat(
            [batch_landmark_nodes, batch_ego_to_landmark_edges, landmark_type, landmark_valid_f], dim=-1
        )  # [B, max_L, 9]
        slot_feats = torch.cat([explore_slots, landmark_slots], dim=1)  # [B, K+max_L, 9]

        # Safety fallback: avoid an all-invalid categorical row.
        no_valid = ~unified_mask.any(dim=1)
        if no_valid.any():
            unified_mask[no_valid, 0] = True
            slot_feats[no_valid, 0, -1] = 1.0

        return unified_mask, unified_relative_pos, node_type_labels, slot_feats

    def _compute_graph_selection_logits(self,
                                        batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks,
                                        batch_explore_nodes, batch_ego_to_explore_edges,
                                        batch_landmark_nodes, batch_landmark_node_masks,
                                        batch_ego_to_landmark_edges):
        """Graph-attention actor logits over unified slots."""
        B = batch_ego_nodes.size(0)
        K = batch_explore_nodes.size(1)
        max_L = batch_landmark_nodes.size(1)
        total_nodes = K + max_L

        # 1) Node/edge embeddings
        explore_node_feats = self.explore_node_encoder(
            batch_explore_nodes.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]

        landmark_node_feats = self.landmark_node_encoder(
            batch_landmark_nodes.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, max_L, 64]

        landmark_node_agg_feats = self.landmark_team_gat(
            batch_ego_nodes[:, :2],
            batch_teammate_nodes, batch_teammate_masks,
            batch_landmark_nodes, batch_landmark_node_masks, landmark_node_feats
        )  # [B, max_L, 64]

        explore_node_feats = self.explore_node_linear(
            explore_node_feats.view(B * K, -1)
        ).view(B, K, -1)
        explore_node_feats = self.linear_ln(explore_node_feats)

        landmark_node_feats = self.landmark_node_linear(
            landmark_node_agg_feats.view(B * max_L, -1)
        ).view(B, max_L, -1)
        landmark_node_feats = self.linear_ln(landmark_node_feats)

        explore_edge_feats = self.edge_encoder(
            batch_ego_to_explore_edges.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 32]
        landmark_edge_feats = self.edge_encoder(
            batch_ego_to_landmark_edges.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, max_L, 32]

        # 2) Unified slot values
        unified_v = torch.zeros(B, total_nodes, self.attn_dim, device=batch_ego_nodes.device)
        explore_kv = torch.cat([explore_node_feats, explore_edge_feats], dim=-1)  # [B, K, 96]
        unified_v[:, :K, :] = self.v_proj(explore_kv.view(B * K, -1)).view(B, K, -1)

        landmark_kv = torch.cat([landmark_node_feats, landmark_edge_feats], dim=-1)  # [B, max_L, 96]
        unified_v[:, K:K + max_L, :] = self.v_proj(landmark_kv.view(B * max_L, -1)).view(B, max_L, -1)

        return self.node_selection_head(unified_v).squeeze(-1)  # [B, K+max_L]

    def _compute_mlp_selection_logits(self,
                                      batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks, slot_feats):
        """MLP actor logits over fixed-length slots."""
        B, total_nodes = slot_feats.size(0), slot_feats.size(1)
        teammate_with_mask = torch.cat([batch_teammate_nodes, batch_teammate_masks], dim=-1)  # [B, A, 6]
        context_input = torch.cat([batch_ego_nodes, teammate_with_mask.view(B, -1)], dim=-1)  # [B, 5+A*6]
        context_feat = self.mlp_context_encoder(context_input)  # [B, 64]

        slot_feat = self.mlp_slot_encoder(slot_feats.view(B * total_nodes, -1)).view(B, total_nodes, -1)  # [B, T, 64]
        context_expand = context_feat.unsqueeze(1).expand(B, total_nodes, context_feat.size(-1))
        fused = torch.cat([context_expand, slot_feat], dim=-1)  # [B, T, 128]
        return self.mlp_selection_head(fused.view(B * total_nodes, -1)).view(B, total_nodes)  # [B, T]

    def _build_high_selection_distribution(self,
                                           batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks,
                                           batch_explore_nodes, batch_ego_to_explore_edges,
                                           batch_landmark_nodes, batch_landmark_node_masks,
                                           batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks):
        """Build logits/distribution and unified slot metadata for high-level actor."""
        if self.high_actor_backbone == "mlp":
            (
                batch_explore_nodes, batch_ego_to_explore_edges,
                batch_landmark_nodes, batch_landmark_node_masks,
                batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
            ) = self._sort_high_level_nodes_for_mlp(
                batch_explore_nodes, batch_ego_to_explore_edges,
                batch_landmark_nodes, batch_landmark_node_masks,
                batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
            )

        unified_mask, unified_relative_pos, node_type_labels, slot_feats = self._build_unified_slot_tensors(
            batch_explore_nodes, batch_ego_to_explore_edges,
            batch_landmark_nodes, batch_landmark_node_masks,
            batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
        )

        if self.high_actor_backbone == "mlp":
            selection_logits = self._compute_mlp_selection_logits(
                batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks, slot_feats
            )
        else:
            selection_logits = self._compute_graph_selection_logits(
                batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks,
                batch_explore_nodes, batch_ego_to_explore_edges,
                batch_landmark_nodes, batch_landmark_node_masks,
                batch_ego_to_landmark_edges
            )

        selection_logits = selection_logits.masked_fill(~unified_mask, -1e9)
        node_dist = TorchCategorical(logits=selection_logits)
        return selection_logits, node_dist, unified_mask, unified_relative_pos, node_type_labels

    def _map_goals_to_selected_indices(self, goals, tasks, ego_nodes, unified_relative_pos, unified_mask, K):
        """Map stored (goal, task) back to selected slot index."""
        B, total_nodes = unified_mask.size()
        goals_relative = goals - ego_nodes[:, :2]  # [B, 2]
        dists = torch.norm(unified_relative_pos - goals_relative.unsqueeze(1), dim=-1)  # [B, T]
        dists_masked = dists.masked_fill(~unified_mask, float('inf'))

        if tasks is not None:
            node_ids = torch.arange(total_nodes, device=goals.device).unsqueeze(0).expand(B, total_nodes)
            is_landmark_task = tasks.squeeze(-1) > 0.5
            task_domain_mask = torch.where(
                is_landmark_task.unsqueeze(1),
                node_ids >= K,
                node_ids < K
            )
            dists_task = dists_masked.masked_fill(~task_domain_mask, float('inf'))
            no_task_valid = torch.isinf(dists_task).all(dim=1)
            if no_task_valid.any():
                dists_task[no_task_valid] = dists_masked[no_task_valid]
            dists_masked = dists_task

        return torch.argmin(dists_masked, dim=-1)  # [B]

    def get_high_level_goal(self, batch_ego_nodes, 
                            batch_teammate_nodes, batch_teammate_masks,
                            batch_explore_nodes, batch_ego_to_explore_edges, 
                            batch_landmark_nodes, batch_landmark_node_masks, 
                            batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks, 
                            deterministic=False):
        """
        统一从所有候选节点（explore + landmark）中选择一个目标
        
        Args:
            batch_ego_nodes: [B, 5] [x, y, vel_x, vel_y, battery]
            batch_teammate_nodes: [B, num_agents, 5] [x, y, vel_x, vel_y, dist_to_goal]
            batch_teammate_masks: [B, num_agents, 1] (1=有效, 0=无效)
            batch_explore_nodes: [B, K, 4] [relative_x, relative_y, utility, occupied]
            batch_ego_to_explore_edges: [B, K, 3] [d, cosθ, sinθ]
            batch_landmark_nodes: [B, Max_L, 4] [relative_x, relative_y, utility, is_targeted]
            batch_landmark_node_masks: [B, Max_L, 1] (1=有效, 0=无效)
            batch_ego_to_landmark_edges: [B, Max_L, 3]
            batch_ego_to_landmark_edge_masks: [B, Max_L, 1]
            deterministic: bool
        
        Returns:
            dict:
                action_modes: [B, 1] (0=explore, 1=landmark) 被选中节点的类型
                waypoints: [B, 2] 绝对世界坐标
                decision_log_probs: [B, 1] 节点选择的 log_prob
                map_log_probs: [B, 1] 同上（保持接口兼容）
        """
        B = batch_ego_nodes.size(0)
        selection_logits, node_dist, unified_mask, unified_relative_pos, node_type_labels = \
            self._build_high_selection_distribution(
                batch_ego_nodes, batch_teammate_nodes, batch_teammate_masks,
                batch_explore_nodes, batch_ego_to_explore_edges,
                batch_landmark_nodes, batch_landmark_node_masks,
                batch_ego_to_landmark_edges, batch_ego_to_landmark_edge_masks
            )

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
        batch_indices = torch.arange(B, device=batch_ego_nodes.device)
        
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

    def evaluate_high_actions(self, env_states, obs, masks_batch,
                          critic_maps, critic_nodes, goals, tasks, 
                          ego_nodes, explore_nodes, 
                          landmark_datas, landmark_masks, landmark_nodes,
                          teammate_nodes, teammate_masks,
                          agent_ids):
        """
        评估给定高层动作的log_prob、熵和价值（用于PPO更新）
        
        Args:
            env_states: [batch, env_dim]
            obs: [batch, obs_dim] - 用于提取智能体位置
            masks_batch: [batch, 1] - 动作掩码
            critic_maps: [batch, 3, H, W] - 用于critic
            critic_nodes: [batch, num_agents, 4] - 用于critic
            goals: [batch, 2] - 已选择的目标位置（世界坐标）
            tasks: [batch, 1] - 已选择的任务类型（0=explore, 1=landmark）
            ego_nodes: [batch, 5]
            explore_nodes: [batch, K, 4]
            landmark_datas: [batch, num_landmarks, 4]
            landmark_masks: [batch, num_landmarks, 1]
            landmark_nodes: [batch, Max_L, 4]
            teammate_nodes: [batch, num_agents, 5]
            teammate_masks: [batch, num_agents, 1]
            agent_ids: [batch, 1] - 智能体ID
            
        Returns:
            high_values: [batch, 1] - 状态价值
            node_log_probs: [batch, 1] - 给定节点选择的log概率
            node_entropy: [batch, 1] - 节点选择分布的熵
        """
        batch_size = env_states.size(0)
        K = explore_nodes.size(1)
        ego_to_explore_edges, ego_to_landmark_edges, ego_to_landmark_edge_masks = self.get_edge_features(
            explore_nodes=explore_nodes,
            landmark_nodes=landmark_nodes,
            landmark_node_masks=landmark_masks,
            norm=False,
            max_distance=2.8
        )
        _, node_dist, unified_mask, unified_relative_pos, _ = self._build_high_selection_distribution(
            ego_nodes, teammate_nodes, teammate_masks,
            explore_nodes, ego_to_explore_edges,
            landmark_nodes, landmark_masks,
            ego_to_landmark_edges, ego_to_landmark_edge_masks
        )
        selected_idx = self._map_goals_to_selected_indices(
            goals, tasks, ego_nodes, unified_relative_pos, unified_mask, K
        )
        node_log_probs = node_dist.log_prob(selected_idx).unsqueeze(-1)  # [B, 1]
        node_entropy = node_dist.entropy().unsqueeze(-1)  # [B, 1]
        
        # =====================================================
        # 5. 计算 Critic 价值
        # =====================================================
        # 计算所有智能体的价值 [B, num_agents]
        all_values = self.get_high_value(critic_maps, critic_nodes)  # [B, num_agents]
        
        # 根据 agent_ids 选择对应的价值
        agent_ids_flat = agent_ids.squeeze(-1)  # [B]
        batch_indices = torch.arange(batch_size, device=env_states.device)
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
