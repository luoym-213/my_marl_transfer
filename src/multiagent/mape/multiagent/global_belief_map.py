"""
Global Belief Map for tracking belief states in multi-agent environments.
Uses Bayesian updates for belief propagation.
"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt, label, center_of_mass

class GlobalBeliefMap:
    """
    基于贝叶斯更新的全局信念地图，用于跟踪团队对环境的信念状态。
    """
    
    def __init__(self, world_size=2.0, cell_size=0.02, initial_belief=0.5, sensor_fidelity=0.8, landmark_positions=None, landmark_radius=0.05):
        self.world_size = world_size
        self.cell_size = cell_size
        self.initial_belief = initial_belief
        self.sensor_fidelity = sensor_fidelity  # p_s
        self.landmark_positions = landmark_positions if landmark_positions is not None else []

        # 计算地图维度: 2 / 0.02 = 100
        self.map_dim = int(world_size / cell_size)
        self.belief_grid = np.full((self.map_dim, self.map_dim), initial_belief, dtype=np.float32)
        
        self.world_min = -world_size / 2.0
        self.world_max = world_size / 2.0
        
        self._precompute_cell_centers()

        self.landmark_map = np.zeros((self.map_dim, self.map_dim), dtype=bool)
        for lx, ly in self.landmark_positions:
            dist_sq = (self.cell_world_x - lx)**2 + (self.cell_world_y - ly)**2
            self.landmark_map |= (dist_sq <= landmark_radius**2)

        self.epsilon = 1e-10
        
    def _precompute_cell_centers(self):
        """预计算每个栅格中心点的世界坐标"""
        grid_x, grid_y = np.meshgrid(
            np.arange(self.map_dim),
            np.arange(self.map_dim),
            indexing='ij'
        )
        
        self.cell_world_x = self.world_min + (grid_x + 0.5) * self.cell_size
        self.cell_world_y = self.world_min + (grid_y + 0.5) * self.cell_size
        
    def reset(self):
        """重置地图，将所有栅格的信念恢复为初始值"""
        self.belief_grid.fill(self.initial_belief)
        
    def world_to_grid(self, world_pos):
        """将世界坐标转换为栅格索引"""
        x, y = world_pos
        
        if not (self.world_min <= x <= self.world_max and 
                self.world_min <= y <= self.world_max):
            return None
            
        i = int((x - self.world_min) / self.cell_size)
        j = int((y - self.world_min) / self.cell_size)
        
        i = np.clip(i, 0, self.map_dim - 1)
        j = np.clip(j, 0, self.map_dim - 1)
        
        return (i, j)
    
    def get_fov_mask(self, agent_pos, obs_radius):
        """计算智能体的观测范围掩码"""
        x, y = agent_pos
        dist_sq = (self.cell_world_x - x)**2 + (self.cell_world_y - y)**2
        return dist_sq <= obs_radius**2
    
    def bayesian_update(self, positive_mask, negative_mask):
        """使用贝叶斯规则更新信念"""
        # 正向更新：检测到目标
        if np.any(positive_mask):
            b_prev = self.belief_grid[positive_mask]
            p_s = self.sensor_fidelity
            
            numerator = p_s * b_prev
            denominator = p_s * b_prev + (1 - p_s) * (1 - b_prev)
            denominator = np.maximum(denominator, self.epsilon)
            
            b_new = numerator / denominator
            b_new = np.clip(b_new, 0.0, 1.0)
            self.belief_grid[positive_mask] = b_new
        
        # 负向更新：未检测到目标
        if np.any(negative_mask):
            b_prev = self.belief_grid[negative_mask]
            p_s = self.sensor_fidelity
            
            numerator = (1 - p_s) * b_prev
            denominator = (1 - p_s) * b_prev + p_s * (1 - b_prev)
            denominator = np.maximum(denominator, self.epsilon)
            
            b_new = numerator / denominator
            b_new = np.clip(b_new, 0.0, 1.0)
            self.belief_grid[negative_mask] = b_new
    
    def update_beliefs(self, agent_positions, obs_radius):
        """
        根据智能体观测和 landmark 实际位置更新信念地图
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            obs_radius: 观测半径
        """
        # 1. 计算所有智能体的观测范围（FOV）
        fov_mask = np.zeros((self.map_dim, self.map_dim), dtype=bool)
        for agent_pos in agent_positions:
            mask = self.get_fov_mask(agent_pos, obs_radius)
            fov_mask |= mask
        
        # 2. 正向更新：被观测到 且 包含 landmark 的区域
        positive_mask = fov_mask & self.landmark_map
        # 3. 负向更新：被观测到 但 不包含 landmark 的区域
        negative_mask = fov_mask & (~self.landmark_map)

        # 4. 执行贝叶斯更新
        self.bayesian_update(positive_mask, negative_mask)
    
    def compute_shannon_entropy(self):
        """计算每个栅格的香农熵"""
        b = self.belief_grid
        
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.where(
                (b > self.epsilon) & (b < 1 - self.epsilon),
                -(b * np.log2(b) + (1 - b) * np.log2(1 - b)),
                0.0
            )
        
        return entropy
    
    def get_total_uncertainty(self):
        """计算整个地图的总不确定性"""
        entropy = self.compute_shannon_entropy()
        return np.sum(entropy)
    
    def get_mean_uncertainty(self):
        """计算整个地图的平均不确定性"""
        entropy = self.compute_shannon_entropy()
        return np.mean(entropy)
    
    def get_high_uncertainty_positions(self, threshold=0.9):
        """获取高不确定性区域的位置"""
        entropy = self.compute_shannon_entropy()
        max_entropy = 1.0
        
        high_uncertainty_mask = entropy >= (threshold * max_entropy)
        indices = np.argwhere(high_uncertainty_mask)
        
        return [(int(i), int(j)) for i, j in indices]
    
    def grid_to_world(self, grid_pos):
        """将栅格索引转换为世界坐标"""
        i, j = grid_pos
        x = self.world_min + (i + 0.5) * self.cell_size
        y = self.world_min + (j + 0.5) * self.cell_size
        return (x, y)
    
    def get_belief_at_position(self, world_pos):
        """获取指定世界坐标处的信念值"""
        grid_pos = self.world_to_grid(world_pos)
        if grid_pos is None:
            return None
        
        i, j = grid_pos
        return self.belief_grid[i, j]
    
    def visualize_beliefs(self):
        """返回信念地图的副本"""
        return self.belief_grid.copy()
    
    def visualize_entropy(self):
        """返回熵地图的副本"""
        return self.compute_shannon_entropy()
    
    def compute_voronoi_regions(self, agent_positions):
        """
        基于智能体位置计算Voronoi区域划分（向量化优化版本）
        """
        if len(agent_positions) == 0:
            return None
        
        # 将智能体位置转换为栅格索引
        agent_grids = []
        for pos in agent_positions:
            grid_pos = self.world_to_grid(pos)
            if grid_pos is not None:
                agent_grids.append(grid_pos)
            else:
                x, y = pos
                x = np.clip(x, self.world_min, self.world_max)
                y = np.clip(y, self.world_min, self.world_max)
                grid_pos = self.world_to_grid((x, y))
                agent_grids.append(grid_pos)
        
        agent_grids = np.array(agent_grids)
        
        # 🚀 向量化优化：创建栅格网格
        grid_i, grid_j = np.meshgrid(
            np.arange(self.map_dim), 
            np.arange(self.map_dim), 
            indexing='ij'
        )
        
        # 🚀 向量化优化：为每个智能体计算到所有栅格的距离
        voronoi_map = np.zeros((self.map_dim, self.map_dim), dtype=np.int32)
        min_dist_map = np.full((self.map_dim, self.map_dim), np.inf)
        
        for agent_idx, (ai, aj) in enumerate(agent_grids):
            # 向量化距离计算
            dist_map = np.sqrt((grid_i - ai)**2 + (grid_j - aj)**2)
            
            # 更新 Voronoi 区域
            mask = dist_map < min_dist_map
            voronoi_map[mask] = agent_idx
            min_dist_map[mask] = dist_map[mask]
        
        return voronoi_map
    
    def compute_entropy_weighted_centroids(self, agent_positions):
        """
        计算每个智能体Voronoi区域的香农熵加权质心（向量化优化版本）
        """
        if len(agent_positions) == 0:
            return []
        
        voronoi_map = self.compute_voronoi_regions(agent_positions)
        entropy_map = self.compute_shannon_entropy()
        
        centroids = []
        
        for agent_idx in range(len(agent_positions)):
            region_mask = (voronoi_map == agent_idx)
            
            # 🚀 向量化优化：直接使用预计算的世界坐标数组
            region_world_x = self.cell_world_x[region_mask]
            region_world_y = self.cell_world_y[region_mask]
            region_entropies = entropy_map[region_mask]
            
            if len(region_entropies) == 0:
                centroids.append(agent_positions[agent_idx])
                continue
            
            total_entropy = np.sum(region_entropies)
            
            if total_entropy < self.epsilon:
                centroids.append(agent_positions[agent_idx])
                continue
            
            # 🚀 向量化优化：一次性计算加权质心
            centroid_x = np.sum(region_world_x * region_entropies) / total_entropy
            centroid_y = np.sum(region_world_y * region_entropies) / total_entropy
            
            centroids.append((float(centroid_x), float(centroid_y)))
    
        return centroids
    
    def get_voronoi_region_stats(self, agent_positions):
        """获取每个智能体Voronoi区域的统计信息"""
        voronoi_map = self.compute_voronoi_regions(agent_positions)
        entropy_map = self.compute_shannon_entropy()
        centroids = self.compute_entropy_weighted_centroids(agent_positions)
        
        stats = []
        
        for agent_idx in range(len(agent_positions)):
            region_mask = (voronoi_map == agent_idx)
            
            area = np.sum(region_mask)
            total_entropy = np.sum(entropy_map[region_mask])
            mean_entropy = np.mean(entropy_map[region_mask]) if area > 0 else 0.0
            mean_belief = np.mean(self.belief_grid[region_mask]) if area > 0 else 0.5
            
            stats.append({
                'agent_idx': agent_idx,
                'area': int(area),
                'total_entropy': float(total_entropy),
                'mean_entropy': float(mean_entropy),
                'centroid': centroids[agent_idx],
                'mean_belief': float(mean_belief)
            })
        
        return stats
    
    def detect_targets(self, belief_threshold=0.95):
        """检测信念地图中的目标点"""
        binary_map = (self.belief_grid > belief_threshold).astype(np.int8)
        
        structure = np.ones((3, 3), dtype=np.int8)
        cluster_labels, num_clusters = label(binary_map, structure=structure)
        
        target_positions = []
        target_grid_positions = []
        cluster_sizes = []
        
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = (cluster_labels == cluster_id)
            size = np.sum(cluster_mask)
            cluster_sizes.append(size)
            
            grid_centroid = center_of_mass(cluster_mask)
            i_center, j_center = int(round(grid_centroid[0])), int(round(grid_centroid[1]))
            
            i_center = np.clip(i_center, 0, self.map_dim - 1)
            j_center = np.clip(j_center, 0, self.map_dim - 1)
            
            target_grid_positions.append((i_center, j_center))
            world_pos = self.grid_to_world((i_center, j_center))
            target_positions.append(world_pos)
        
        return {
            'binary_map': binary_map,
            'num_targets': num_clusters,
            'target_positions': target_positions,
            'target_grid_positions': target_grid_positions,
            'cluster_sizes': cluster_sizes,
            'cluster_labels': cluster_labels
        }
    
    def get_target_positions(self, belief_threshold=0.95, min_cluster_size=1):
        """直接返回检测到的目标点世界坐标"""
        result = self.detect_targets(belief_threshold)
        
        if min_cluster_size > 1:
            filtered_positions = []
            for pos, size in zip(result['target_positions'], result['cluster_sizes']):
                if size >= min_cluster_size:
                    filtered_positions.append(pos)
            return filtered_positions
        
        return result['target_positions']
    
    def visualize_detected_targets(self, belief_threshold=0.95):
        """可视化检测到的目标点"""
        result = self.detect_targets(belief_threshold)
        
        vis_map = result['cluster_labels'].astype(np.float32)
        
        for grid_pos in result['target_grid_positions']:
            i, j = grid_pos
            vis_map[i, j] = -1
        
        result['visualization_map'] = vis_map
        return result
    
    def get_targets_summary(self, belief_threshold=0.95):
        """获取目标检测的摘要信息"""
        result = self.detect_targets(belief_threshold)
        
        summary = {
            'num_targets': result['num_targets'],
            'total_high_belief_cells': np.sum(result['binary_map']),
            'mean_cluster_size': np.mean(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'max_cluster_size': max(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'min_cluster_size': min(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'target_positions': result['target_positions']
        }
        
        return summary

