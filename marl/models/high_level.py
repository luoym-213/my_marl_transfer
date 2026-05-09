import math

import torch
from torch.distributions import Categorical as TorchCategorical

from marl.controllers.high_level_policy import (
    build_edge_features,
    build_explore_nodes,
    build_landmark_nodes,
    grid_to_world_torch,
    world_to_grid_torch,
)


class HighLevel:
    def get_explore_nodes(self, top_k, rrt_max_iter, vec_inp, map_inp, agent_indices=None, deterministic=False):
        return build_explore_nodes(
            top_k,
            rrt_max_iter,
            vec_inp,
            map_inp,
            agent_indices=agent_indices,
        )

    def get_landmark_nodes(self, agent_positions, detected, detected_mask, linear_indices, all_masks=None):
        return build_landmark_nodes(
            agent_positions,
            detected,
            detected_mask,
            linear_indices,
            num_agents=self.num_agents,
            all_masks=all_masks,
        )

    def get_edge_features(self, explore_nodes, landmark_nodes, landmark_node_masks, norm=False, max_distance=2.8):
        return build_edge_features(
            explore_nodes,
            landmark_nodes,
            landmark_node_masks,
            norm=norm,
            max_distance=max_distance,
        )

    def _world_to_grid_torch(self, world_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return world_to_grid_torch(world_xy, H, W)

    def _grid_to_world_torch(self, grid_ij: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return grid_to_world_torch(grid_ij, H, W)

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
        K = batch_explore_nodes.size(1)
        
        # ===== 1. 节点和边的特征嵌入 =====
        # 1.1 Ego节点嵌入
        ego_node_feats = self.ego_node_encoder(batch_ego_nodes)  # [B, 64]
        
        # 1.2 Explore节点嵌入
        explore_node_feats = self.explore_node_encoder(
            batch_explore_nodes.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]
 
        # 1.3 Landmark节点GAT聚合
        max_L = batch_landmark_nodes.size(1)  # [B, Max_L, 4]

        # 一次性编码所有 landmark 节点 [B, Max_L, 4] -> [B, Max_L, 64]
        landmark_node_feats = self.landmark_node_encoder(
            batch_landmark_nodes.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 64]

        # landmark 和 队友节点进行聚合
        landmark_node_agg_feats = self.landmark_team_gat(
            batch_ego_nodes[:, :2],  # 只传递 ego 的位置用于计算相对位置
            batch_teammate_nodes, batch_teammate_masks,
            batch_landmark_nodes, batch_landmark_node_masks, landmark_node_feats
        )  # [B, Max_L, 64]

        # 1.4 对explore和landmark节点特征进行线性变换和LayerNorm
        explore_node_feats = self.explore_node_linear(
            explore_node_feats.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]

        explore_node_feats = self.linear_ln(explore_node_feats)  # LayerNorm

        landmark_node_feats = self.landmark_node_linear(
            landmark_node_agg_feats.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 64]

        landmark_node_feats = self.linear_ln(landmark_node_feats)  # LayerNorm

        # 1.5 编码所有 landmark 边 [B, Max_L, 3] -> [B, Max_L, 32]
        landmark_edge_feats = self.edge_encoder(
            batch_ego_to_landmark_edges.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 32]

        # 编码 explore 边 [B, K, 3] -> [B, K, 32]
        explore_edge_feats = self.edge_encoder(
            batch_ego_to_explore_edges.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 32]

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
        # 拼接节点和边特征 [B, Max_L, 64] + [B, Max_L, 32] -> [B, Max_L, 96]
        lm_kv = torch.cat([landmark_node_feats, landmark_edge_feats], dim=-1)
        # 投影为 K/V [B, Max_L, 96] -> [B, Max_L, 64]
        unified_k[:, K:K+max_L, :] = self.k_proj(lm_kv.view(B * max_L, -1)).view(B, max_L, -1)
        unified_v[:, K:K+max_L, :] = self.v_proj(lm_kv.view(B * max_L, -1)).view(B, max_L, -1)
        
        # 构建有效掩码：mask 有效 且 未被追踪
        valid_mask = batch_landmark_node_masks[:, :, 0] > 0.5  # [B, Max_L]
        not_targeted = batch_landmark_nodes[:, :, 3] < 0.5    # [B, Max_L]
        combined_mask = valid_mask & not_targeted              # [B, Max_L]

        # 如果valid_mask某行全部有效，则说明已经找到全部landmark，则该样本不需要探索节点
        all_landmarks_found = valid_mask.all(dim=1)  # [B] bool tensor
        if all_landmarks_found.any():
            unified_mask[all_landmarks_found, :K] = False
        
        unified_mask[:, K:K+max_L] = combined_mask
        unified_relative_pos[:, K:K+max_L, :] = batch_landmark_nodes[:, :, :2]
        node_type_labels[:, K:K+max_L] = 1

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
        num_agents = self.num_agents
        K = explore_nodes.size(1)  # explore节点数量

        # =====================================================
        # 1. 重建Graph nodes和Edges
        # =====================================================

        # 计算边特征
        ego_to_explore_edges, ego_to_landmark_edges, ego_to_landmark_edge_masks = self.get_edge_features(
            explore_nodes=explore_nodes,
            landmark_nodes=landmark_nodes,
            landmark_node_masks=landmark_masks,
            norm=False,
            max_distance=2.8
        )

        # =====================================================
        # 2. 复用 get_high_level_goal 的逻辑构建节点分布
        # =====================================================
        B = batch_size
        
        # 2.1 节点和边的特征嵌入
        ego_node_feats = self.ego_node_encoder(ego_nodes)  # [B, 64]
        
        # 2.2 Explore节点嵌入
        explore_node_feats = self.explore_node_encoder(
            explore_nodes.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]
        
        # 2.3 Landmark节点GAT聚合
        max_L = landmark_nodes.size(1)  # [B, Max_L, 4]

        # 一次性编码所有 landmark 节点 [B, Max_L, 4] -> [B, Max_L, 64]
        landmark_node_feats = self.landmark_node_encoder(
            landmark_nodes.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 64]

        # landmark 和 队友节点进行聚合
        landmark_node_agg_feats = self.landmark_team_gat(
            ego_nodes[:, :2],  # 只传递 ego 的位置用于计算相对位置
            teammate_nodes, teammate_masks,
            landmark_nodes, landmark_masks, landmark_node_feats
        )  # [B, Max_L, 64]

        # 2.4 对explore和landmark节点特征进行线性变换和LayerNorm
        explore_node_feats = self.explore_node_linear(
            explore_node_feats.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 64]

        explore_node_feats = self.linear_ln(explore_node_feats)  # LayerNorm

        landmark_node_feats = self.landmark_node_linear(
            landmark_node_agg_feats.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 64]

        landmark_node_feats = self.linear_ln(landmark_node_feats)  # LayerNorm

        # 2.5 一次性编码所有 landmark 边 [B, Max_L, 3] -> [B, Max_L, 32]
        landmark_edge_feats = self.edge_encoder(
            ego_to_landmark_edges.view(B * max_L, -1)
        ).view(B, max_L, -1)  # [B, Max_L, 32]

        explore_edge_feats = self.edge_encoder(
            ego_to_explore_edges.view(B * K, -1)
        ).view(B, K, -1)  # [B, K, 32]
        
        # 2.2 构建统一的候选节点集合
        total_nodes = K + max_L
        
        q = self.q_proj(ego_node_feats).unsqueeze(1)  # [B, 1, 64]
        
        unified_k = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_v = torch.zeros(B, total_nodes, self.attn_dim, device=ego_node_feats.device)
        unified_mask = torch.zeros(B, total_nodes, device=ego_node_feats.device, dtype=torch.bool)
        unified_relative_pos = torch.zeros(B, total_nodes, 2, device=ego_node_feats.device)

        # 节点类型标签: 0=explore, 1=landmark
        node_type_labels = torch.zeros(B, total_nodes, device=ego_node_feats.device, dtype=torch.long)
        
        # 填充 explore 节点
        explore_kv = torch.cat([explore_node_feats, explore_edge_feats], dim=-1)
        unified_k[:, :K, :] = self.k_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_v[:, :K, :] = self.v_proj(explore_kv.view(B * K, -1)).view(B, K, -1)
        unified_mask[:, :K] = True  # explore 节点全部有效
        unified_relative_pos[:, :K, :] = explore_nodes[:, :, :2]  # 相对坐标
        node_type_labels[:, :K] = 0  # explore 类型
        
        # 填充 landmark 节点 (索引 K ~ K+max_L-1)
        # 拼接节点和边特征 [B, Max_L, 64] + [B, Max_L, 32] -> [B, Max_L, 96]
        lm_kv = torch.cat([landmark_node_feats, landmark_edge_feats], dim=-1)
        # 投影为 K/V [B, Max_L, 96] -> [B, Max_L, 64]
        unified_k[:, K:K+max_L, :] = self.k_proj(lm_kv.view(B * max_L, -1)).view(B, max_L, -1)
        unified_v[:, K:K+max_L, :] = self.v_proj(lm_kv.view(B * max_L, -1)).view(B, max_L, -1)
        
        # 构建有效掩码：mask 有效 且 未被追踪
        valid_mask = landmark_masks[:, :, 0] > 0.5  # [B, Max_L]
        not_targeted = landmark_nodes[:, :, 3] < 0.5    # [B, Max_L]
        combined_mask = valid_mask & not_targeted    # [B, Max_L]
        
        # 如果valid_mask某行全部有效，则说明已经找到全部landmark，则该样本不需要探索节点
        all_landmarks_found = valid_mask.all(dim=1)  # [B] bool tensor
        if all_landmarks_found.any():
            unified_mask[all_landmarks_found, :K] = False          
        
        unified_mask[:, K:K+max_L] = combined_mask
        unified_relative_pos[:, K:K+max_L, :] = landmark_nodes[:, :, :2]
        node_type_labels[:, K:K+max_L] = 1
        
        # 3. 注意力机制
        scale = math.sqrt(self.attn_dim)
        attn_scores = (q @ unified_k.transpose(1, 2)) / scale
        attn_scores = attn_scores.masked_fill(~unified_mask.unsqueeze(1), -1e9)
        
        # Softmax 得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, total_nodes]
        
        # 注意力加权求和
        context = (attn_weights @ unified_v).squeeze(1)  # [B, 64]
        
        # 2.4 节点选择分布
        selection_logits = self.node_selection_head(unified_v).squeeze(-1)  # [B, total_nodes]
        
        # 应用 mask
        selection_logits = selection_logits.masked_fill(~unified_mask, -1e9)
        
        # 构建分类分布
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
