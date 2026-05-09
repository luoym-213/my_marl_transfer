import numpy as np
import torch.nn as nn

from marl.algos.distributions import Categorical

from marl.models.accessors import ModuleAccessors
from marl.models.checkpointing import Checkpointing
from marl.models.common import weights_init
from marl.models.high_level import HighLevel
from marl.models.low_level import LowLevelPolicy


class MPNN(
    nn.Module,
    Checkpointing,
    ModuleAccessors,
    HighLevel,
    LowLevelPolicy,
):
    def __init__(
        self,
        action_space,
        num_agents,
        num_entities,
        input_size=16,
        hidden_dim=128,
        embed_dim=None,
        pos_index=2,
        norm_in=False,
        nonlin=nn.ReLU,
        n_heads=3,
        mask_dist=None,
        mask_obs_dist=None,
        entity_mp=False,
        is_recurrent=True,
    ):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents
        self.num_entities = num_entities
        self.low_level_input = 2 + 2 * num_agents
        self.K = 3
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.is_recurrent = is_recurrent
        self.mask_dist = mask_dist
        self.mask_obs_dist = mask_obs_dist
        self.input_size = input_size
        self.entity_mp = entity_mp
        self.pos_index = pos_index
        self.task_dim = 2
        self.h_dim2 = self.h_dim // 2
        num_actions = action_space.n

        self.modules_dict = nn.ModuleDict()
        self.modules_dict["low_level"] = self._build_low_level_modules(action_space)
        self.modules_dict["high_level"] = self._build_high_level_modules()
        self.modules_dict["high_critic"] = self._build_high_critic_modules()

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.apply(weights_init)
        self.attn_mat = np.ones((num_agents, num_agents))
        self.dropout_mask = None

        # Preserve the original extra registered distribution module.
        self.dist = Categorical(self.h_dim, num_actions)

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
