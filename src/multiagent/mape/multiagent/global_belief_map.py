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
    
    参数:
        world_size: 世界的总尺寸 (从 -world_size/2 到 +world_size/2)
        cell_size: 每个栅格的尺寸
        initial_belief: 初始信念值 (默认 0.5)
        sensor_fidelity: 传感器探测保真度 p_s (默认 0.8)
    """
    
    def __init__(self, world_size=2.0, cell_size=0.02, initial_belief=0.5, sensor_fidelity=0.8, landmark_positions=None, landmark_radius=0.05):
        self.world_size = world_size
        self.cell_size = cell_size
        self.initial_belief = initial_belief
        self.sensor_fidelity = sensor_fidelity  # p_s
        self.landmark_positions = landmark_positions if landmark_positions is not None else []

        # 计算地图维度: 2 / 0.02 = 100
        self.map_dim = int(world_size / cell_size)
        
        # 初始化信念地图 (所有栅格初始信念为 0.5)
        self.belief_grid = np.full((self.map_dim, self.map_dim), initial_belief, dtype=np.float32)
        
        # 世界坐标的边界(-1, 1)
        self.world_min = -world_size / 2.0
        self.world_max = world_size / 2.0
        
        # 预计算每个栅格中心的世界坐标
        self._precompute_cell_centers()

        # 预计算 landmark 地图，栅格中心点在 landmark 半径内的设为 True
        self.landmark_map = np.zeros((self.map_dim, self.map_dim), dtype=bool)
        for lx, ly in self.landmark_positions:
            dist_sq = (self.cell_world_x - lx)**2 + (self.cell_world_y - ly)**2
            self.landmark_map |= (dist_sq <= landmark_radius**2)

        # 用于防止数值不稳定的小常数
        self.epsilon = 1e-10
        
    def _precompute_cell_centers(self):
        """预计算每个栅格中心点的世界坐标"""
        # 创建栅格索引
        grid_x, grid_y = np.meshgrid(
            np.arange(self.map_dim),
            np.arange(self.map_dim),
            indexing='ij'
        )
        
        # 转换为世界坐标
        self.cell_world_x = self.world_min + (grid_x + 0.5) * self.cell_size
        self.cell_world_y = self.world_min + (grid_y + 0.5) * self.cell_size
        
    def reset(self):
        """重置地图，将所有栅格的信念恢复为初始值"""
        self.belief_grid.fill(self.initial_belief)
        
    def world_to_grid(self, world_pos):
        """
        将世界坐标转换为栅格索引
        
        参数:
            world_pos: (x, y) 世界坐标
            
        返回:
            (i, j) 栅格索引，如果超出边界则返回 None
        """
        x, y = world_pos
        
        # 检查边界
        if not (self.world_min <= x <= self.world_max and 
                self.world_min <= y <= self.world_max):
            return None
            
        # 转换为栅格索引
        i = int((x - self.world_min) / self.cell_size)
        j = int((y - self.world_min) / self.cell_size)
        
        # 确保索引在有效范围内
        i = np.clip(i, 0, self.map_dim - 1)
        j = np.clip(j, 0, self.map_dim - 1)
        
        return (i, j)
    
    def get_fov_mask(self, agent_pos, obs_radius):
        """
        计算智能体的观测范围掩码 (Field of View)
        
        参数:
            agent_pos: (x, y) 智能体的世界坐标
            obs_radius: 观测半径
            
        返回:
            (map_dim, map_dim) 的布尔数组，True 表示在观测范围内
        """
        x, y = agent_pos
        dist_sq = (self.cell_world_x - x)**2 + (self.cell_world_y - y)**2
        return dist_sq <= obs_radius**2
    
    def bayesian_update(self, positive_mask, negative_mask):
        """
        使用贝叶斯规则更新信念
        
        正向更新（检测到目标）: b_{x,y}(t) = [p_s * b_{x,y}(t-1)] / [p_s * b_{x,y}(t-1) + (1-p_s) * (1-b_{x,y}(t-1))]
        负向更新（未检测到目标）: b_{x,y}(t) = [(1-p_s) * b_{x,y}(t-1)] / [(1-p_s) * b_{x,y}(t-1) + p_s * (1-b_{x,y}(t-1))]
        
        参数:
            positive_mask: (map_dim, map_dim) 布尔数组，True 表示该区域被观测到且检测到 landmark
            negative_mask: (map_dim, map_dim) 布尔数组，True 表示该区域被观测到但未检测到 landmark
        """
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
        """
        计算每个栅格的香农熵
        
        香农熵公式: H(b) = -b*log(b) - (1-b)*log(1-b)
        
        返回:
            (map_dim, map_dim) 的熵值数组
        """
        b = self.belief_grid
        
        # 使用 errstate 抑制警告
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.where(
                (b > self.epsilon) & (b < 1 - self.epsilon),
                -(b * np.log2(b) + (1 - b) * np.log2(1 - b)),
                0.0
            )
        
        return entropy
    
    def get_total_uncertainty(self):
        """
        计算整个地图的总不确定性（熵的总和）
        
        返回:
            总不确定性值
        """
        entropy = self.compute_shannon_entropy()
        return np.sum(entropy)
    
    def get_mean_uncertainty(self):
        """
        计算整个地图的平均不确定性
        
        返回:
            平均不确定性值
        """
        entropy = self.compute_shannon_entropy()
        return np.mean(entropy)
    
    def get_high_uncertainty_positions(self, threshold=0.9):
        """
        获取高不确定性区域的位置
        
        参数:
            threshold: 熵阈值（0-1之间），默认0.9
            
        返回:
            高不确定性栅格的索引列表 [(i1, j1), (i2, j2), ...]
        """
        entropy = self.compute_shannon_entropy()
        max_entropy = 1.0  # 二元信念的最大熵
        
        high_uncertainty_mask = entropy >= (threshold * max_entropy)
        indices = np.argwhere(high_uncertainty_mask)
        
        return [(int(i), int(j)) for i, j in indices]
    
    def grid_to_world(self, grid_pos):
        """
        将栅格索引转换为世界坐标（栅格中心）
        
        参数:
            grid_pos: (i, j) 栅格索引
            
        返回:
            (x, y) 世界坐标
        """
        i, j = grid_pos
        x = self.world_min + (i + 0.5) * self.cell_size
        y = self.world_min + (j + 0.5) * self.cell_size
        return (x, y)
    
    def get_belief_at_position(self, world_pos):
        """
        获取指定世界坐标处的信念值
        
        参数:
            world_pos: (x, y) 世界坐标
            
        返回:
            信念值，如果超出边界则返回 None
        """
        grid_pos = self.world_to_grid(world_pos)
        if grid_pos is None:
            return None
        
        i, j = grid_pos
        return self.belief_grid[i, j]
    
    def visualize_beliefs(self):
        """
        返回信念地图的副本（用于可视化）
        
        返回:
            (map_dim, map_dim) 的信念值数组
        """
        return self.belief_grid.copy()
    
    def visualize_entropy(self):
        """
        返回熵地图的副本（用于可视化不确定性）
        
        返回:
            (map_dim, map_dim) 的熵值数组
        """
        return self.compute_shannon_entropy()
    
    def compute_voronoi_regions(self, agent_positions):
        """
        基于智能体位置计算Voronoi区域划分
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            
        返回:
            (map_dim, map_dim) 的整数数组，每个值表示该栅格属于哪个智能体
            值为智能体索引 (0, 1, 2, ...)
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
                # 如果位置超出边界，使用最近的边界点
                x, y = pos
                x = np.clip(x, self.world_min, self.world_max)
                y = np.clip(y, self.world_min, self.world_max)
                grid_pos = self.world_to_grid((x, y))
                agent_grids.append(grid_pos)
        
        # 创建Voronoi区域地图
        voronoi_map = np.zeros((self.map_dim, self.map_dim), dtype=np.int32)
        
        # 为每个栅格计算最近的智能体
        for i in range(self.map_dim):
            for j in range(self.map_dim):
                min_dist = float('inf')
                nearest_agent = 0
                
                for agent_idx, (ai, aj) in enumerate(agent_grids):
                    dist = np.sqrt((i - ai)**2 + (j - aj)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_agent = agent_idx
                
                voronoi_map[i, j] = nearest_agent
        
        return voronoi_map
    
    def compute_entropy_weighted_centroids(self, agent_positions):
        """
        计算每个智能体Voronoi区域的香农熵加权质心
        
        公式: m_i = (Σ_{j∈V(u_i)} x_j*H_j / Σ_{j∈V(u_i)} H_j, 
                     Σ_{j∈V(u_i)} y_j*H_j / Σ_{j∈V(u_i)} H_j)
        
        其中:
        - V(u_i): 智能体 i 的Voronoi区域
        - H_j: 栅格 j 的香农熵
        - (x_j, y_j): 栅格 j 的世界坐标
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            
        返回:
            加权质心列表 [(mx1, my1), (mx2, my2), ...]
        """
        if len(agent_positions) == 0:
            return []
        
        # 计算Voronoi区域划分
        voronoi_map = self.compute_voronoi_regions(agent_positions)
        
        # 计算香农熵地图
        entropy_map = self.compute_shannon_entropy()
        
        centroids = []
        
        for agent_idx in range(len(agent_positions)):
            # 获取该智能体的Voronoi区域掩码
            region_mask = (voronoi_map == agent_idx)
            
            # 获取区域内的栅格索引
            region_indices = np.argwhere(region_mask)
            
            if len(region_indices) == 0:
                # 如果区域为空，返回智能体当前位置
                centroids.append(agent_positions[agent_idx])
                continue
            
            # 提取区域内的香农熵值
            region_entropies = entropy_map[region_mask]
            
            # 计算熵的总和
            total_entropy = np.sum(region_entropies)
            
            if total_entropy < self.epsilon:
                # 如果总熵接近零，返回智能体当前位置
                centroids.append(agent_positions[agent_idx])
                continue
            
            # 计算加权质心
            weighted_x = 0.0
            weighted_y = 0.0
            
            for idx, (i, j) in enumerate(region_indices):
                # 获取栅格中心的世界坐标
                x, y = self.grid_to_world((i, j))
                
                # 获取该栅格的香农熵
                entropy = region_entropies[idx]
                
                # 累加加权坐标
                weighted_x += x * entropy
                weighted_y += y * entropy
            
            # 归一化得到质心
            centroid_x = weighted_x / total_entropy
            centroid_y = weighted_y / total_entropy
            
            centroids.append((centroid_x, centroid_y))
        
        return centroids
    
    def get_voronoi_region_stats(self, agent_positions):
        """
        获取每个智能体Voronoi区域的统计信息
        
        参数:
            agent_positions: 智能体位置列表
            
        返回:
            字典列表，每个字典包含:
            - 'agent_idx': 智能体索引
            - 'area': 区域面积（栅格数）
            - 'total_entropy': 区域总熵
            - 'mean_entropy': 区域平均熵
            - 'centroid': 熵加权质心
            - 'mean_belief': 区域平均信念值
        """
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
        """
        检测信念地图中的目标点
        
        步骤:
        1. 将信念值 > threshold 的栅格标记为 1，其他为 0
        2. 对连续的 1 进行聚类（连通域分析）
        3. 计算每个聚类的质心作为目标点位置
        
        参数:
            belief_threshold: 信念阈值，默认 0.95
            
        返回:
            dict 包含:
            - 'binary_map': (map_dim, map_dim) 二值化地图
            - 'num_targets': 检测到的目标点数量
            - 'target_positions': [(x1, y1), (x2, y2), ...] 目标点的世界坐标
            - 'target_grid_positions': [(i1, j1), (i2, j2), ...] 目标点的栅格坐标
            - 'cluster_sizes': [size1, size2, ...] 每个目标占据的栅格数
            - 'cluster_labels': (map_dim, map_dim) 聚类标签地图
        """
        # 1. 二值化：信念 > threshold 的栅格标记为 1
        binary_map = (self.belief_grid > belief_threshold).astype(np.int8)
        
        # 2. 连通域分析（8-邻域连接）
        # structure 定义连接方式：8-邻域（包括对角线）
        structure = np.ones((3, 3), dtype=np.int8)  # 8-connected
        cluster_labels, num_clusters = label(binary_map, structure=structure)
        
        # 3. 计算每个聚类的质心
        target_positions = []
        target_grid_positions = []
        cluster_sizes = []
        
        for cluster_id in range(1, num_clusters + 1):  # cluster_id 从 1 开始
            # 获取该聚类的掩码
            cluster_mask = (cluster_labels == cluster_id)
            
            # 计算聚类大小
            size = np.sum(cluster_mask)
            cluster_sizes.append(size)
            
            # 计算栅格坐标的质心（平均位置）
            # center_of_mass 返回 (i_center, j_center)
            grid_centroid = center_of_mass(cluster_mask)
            i_center, j_center = int(round(grid_centroid[0])), int(round(grid_centroid[1]))
            
            # 确保索引在有效范围内
            i_center = np.clip(i_center, 0, self.map_dim - 1)
            j_center = np.clip(j_center, 0, self.map_dim - 1)
            
            target_grid_positions.append((i_center, j_center))
            
            # 转换为世界坐标
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
        """
        简化接口：直接返回检测到的目标点世界坐标
        
        参数:
            belief_threshold: 信念阈值
            min_cluster_size: 最小聚类大小（过滤噪声）
            
        返回:
            target_positions: [(x1, y1), (x2, y2), ...] 目标点列表
        """
        result = self.detect_targets(belief_threshold)
        
        # 过滤小聚类（可能是噪声）
        if min_cluster_size > 1:
            filtered_positions = []
            for pos, size in zip(result['target_positions'], result['cluster_sizes']):
                if size >= min_cluster_size:
                    filtered_positions.append(pos)
            return filtered_positions
        
        return result['target_positions']
    
    def visualize_detected_targets(self, belief_threshold=0.95):
        """
        可视化检测到的目标点
        
        参数:
            belief_threshold: 信念阈值
            
        返回:
            dict 包含可视化所需的所有数据
        """
        result = self.detect_targets(belief_threshold)
        
        # 创建可视化地图（在聚类标签上叠加质心标记）
        vis_map = result['cluster_labels'].astype(np.float32)
        
        # 在质心位置标记特殊值
        for grid_pos in result['target_grid_positions']:
            i, j = grid_pos
            vis_map[i, j] = -1  # 用 -1 标记质心
        
        result['visualization_map'] = vis_map
        return result
    
    def get_targets_summary(self, belief_threshold=0.95):
        """
        获取目标检测的摘要信息（用于调试和监控）
        
        参数:
            belief_threshold: 信念阈值
            
        返回:
            字典包含统计信息
        """
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

