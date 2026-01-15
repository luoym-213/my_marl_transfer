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
    
    def __init__(self, world_size=2.0, cell_size=0.02, initial_belief=0.5, sensor_fidelity=0.8, landmark_positions=None, landmark_radius=0.05, obs_radius=0.3):
        self.world_size = world_size
        self.cell_size = cell_size
        self.initial_belief = initial_belief
        self.sensor_fidelity = sensor_fidelity  # p_s
        self.landmark_positions = landmark_positions if landmark_positions is not None else []
        self.landmark_radius = landmark_radius

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
        self.landmark_heatmap = self.get_landmarks_heatmap()

        # 目标发现阈值
        self.belief_threshold = 0.95

        # 设置动态探索奖励归一化分母：根据观测半径、传感器可靠度和地图尺寸预估最大熵值
        # 1. 计算FOV覆盖的栅格数量
        # 使用离散计数代替几何面积，以匹配实际更新的栅格数
        dummy_mask = self.get_fov_mask((0, 0), obs_radius)
        fov_area = np.sum(dummy_mask)
        
        # 2. 计算单次观测单个栅格的最大熵增量
        p_s = self.sensor_fidelity
        
        # 动态计算最大熵减：模拟贝叶斯更新序列，找到最大的单步熵减
        # 解释：熵减在 p=0.5 时不是最大的，通常在 p=0.8 左右进行下一次更新时达到峰值
        max_drop = 0.0
        curr_p = self.initial_belief
        for _ in range(10): # 模拟10次更新足以覆盖峰值
            # 计算当前熵
            h_curr = -(curr_p * np.log2(curr_p) + (1 - curr_p) * np.log2(1 - curr_p))
            
            # 贝叶斯更新
            num = p_s * curr_p
            den = p_s * curr_p + (1 - p_s) * (1 - curr_p)
            next_p = num / den
            
            # 计算更新后熵
            h_next = -(next_p * np.log2(next_p) + (1 - next_p) * np.log2(1 - next_p))
            
            drop = h_curr - h_next
            if drop > max_drop:
                max_drop = drop
            
            curr_p = next_p
            
        max_entropy_increase = max_drop
        # 3. 估计最大可能熵值
        max_possible_entropy = max_entropy_increase * fov_area
        self.explore_reward_normalization = max_possible_entropy if max_possible_entropy > 0 else 1.0

        # 4. 计算最大可能信念增量 (发现奖励归一化)
        # 寻找 max(Bayes(b) - b)，即单格最大可能的信念提升
        self.max_belief_delta = 0.0
        for b in np.linspace(0, 1, 101): # 遍历可能的信念值
            num = p_s * b
            den = p_s * b + (1 - p_s) * (1 - b)
            if den > 1e-10:
                new_b = num / den
                delta = new_b - b
                if delta > self.max_belief_delta:
                    self.max_belief_delta = delta
        
        self._update_discover_normalization()

    def _update_discover_normalization(self):
        """根据当前地图中的landmark总面积更新发现奖励的归一化系数"""
        total_landmark_cells = np.sum(self.landmark_map)
        # 如果没有landmark，使用1.0避免除零（此时奖励也为0）
        normalization_area = total_landmark_cells if total_landmark_cells > 0 else 1.0
        
        max_possible_discover = self.max_belief_delta * normalization_area
        self.discover_reward_normalization = max_possible_discover if max_possible_discover > 0 else 1.0
        
    def _precompute_cell_centers(self):
        """预计算每个栅格中心点的世界坐标"""
        grid_x, grid_y = np.meshgrid(
            np.arange(self.map_dim),
            np.arange(self.map_dim),
            indexing='ij'
        )
        
        self.cell_world_x = self.world_min + (grid_x + 0.5) * self.cell_size
        self.cell_world_y = self.world_min + (grid_y + 0.5) * self.cell_size

    def reset(self, landmark_positions=None):
        """重置地图，将所有栅格的信念恢复为初始值"""
        self.belief_grid.fill(self.initial_belief)

        if landmark_positions is not None:
            self.landmark_positions = landmark_positions
            self.landmark_map = np.zeros((self.map_dim, self.map_dim), dtype=bool)
            for lx, ly in self.landmark_positions:
                dist_sq = (self.cell_world_x - lx)**2 + (self.cell_world_y - ly)**2
                self.landmark_map |= (dist_sq <= self.landmark_radius**2)

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
    
    def compute_voronoi_regions(self, agent_positions, agent_dones=None):
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
        voronoi_map = np.full((self.map_dim, self.map_dim), -1, dtype=np.int32)
        min_dist_map = np.full((self.map_dim, self.map_dim), np.inf)
        
        for agent_idx, (ai, aj) in enumerate(agent_grids):
            if agent_dones is not None and agent_dones[agent_idx]:
               continue

            # 向量化距离计算
            dist_map = np.sqrt((grid_i - ai)**2 + (grid_j - aj)**2)
            
            # 更新 Voronoi 区域
            mask = dist_map < min_dist_map
            voronoi_map[mask] = agent_idx
            min_dist_map[mask] = dist_map[mask]
        
        return voronoi_map
    
    def get_voronoi_edges(self, agent_positions, agent_dones=None):
        """
        获取Voronoi图的边界线段，用于可视化
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            agent_dones: 智能体完成状态列表 [bool, bool, ...] (可选)
        
        返回:
            edges: 边界线段列表 [((x1, y1), (x2, y2)), ...]
        """
        if len(agent_positions) < 2:
            return []
        
        from scipy.spatial import Voronoi
        
        agent_positions = np.array(agent_positions)
        
        # 过滤掉已完成的智能体
        if agent_dones is not None:
            agent_dones = np.array(agent_dones)
            active_mask = ~agent_dones  # 未完成的智能体
            active_positions = agent_positions[active_mask]
            
            # 如果活跃智能体少于2个，无法形成Voronoi图
            if len(active_positions) < 2:
                return []
        else:
            active_positions = agent_positions
        
        boundary = self.world_size / 2.0
        
        # 添加镜像点以获得有限的Voronoi单元
        mirror_points = []
        
        # 添加四个角的镜像点
        corners = [
            [-boundary*3, -boundary*3],
            [-boundary*3, boundary*3],
            [boundary*3, -boundary*3],
            [boundary*3, boundary*3]
        ]
        mirror_points.extend(corners)
        
        # 添加边界上的镜像点（只为活跃智能体添加）
        for pos in active_positions:
            mirror_points.extend([
                [pos[0], boundary*3],      # 上
                [pos[0], -boundary*3],     # 下
                [boundary*3, pos[1]],      # 右
                [-boundary*3, pos[1]]      # 左
            ])
        
        # 合并活跃智能体和镜像点
        all_points = np.vstack([active_positions, mirror_points])
        
        try:
            # 计算Voronoi图
            vor = Voronoi(all_points)
            
            edges = []
            
            # 提取Voronoi边界线段
            for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
                # 只处理有限的边（不包含无穷远点）
                if -1 not in ridge_vertices:
                    # 检查是否至少有一个点是原始活跃智能体
                    if ridge_points[0] < len(active_positions) or ridge_points[1] < len(active_positions):
                        v0 = vor.vertices[ridge_vertices[0]]
                        v1 = vor.vertices[ridge_vertices[1]]
                        
                        # 裁剪到世界边界内
                        v0_clipped = np.clip(v0, -boundary, boundary)
                        v1_clipped = np.clip(v1, -boundary, boundary)
                        
                        # 检查线段是否在边界内
                        if (abs(v0_clipped[0]) <= boundary and abs(v0_clipped[1]) <= boundary and
                            abs(v1_clipped[0]) <= boundary and abs(v1_clipped[1]) <= boundary):
                            
                            edges.append((
                                (float(v0_clipped[0]), float(v0_clipped[1])),
                                (float(v1_clipped[0]), float(v1_clipped[1]))
                            ))
        
            return edges
            
        except Exception as e:
            # 如果Voronoi计算失败，返回空列表
            return []
    
    def get_voronoi_region_masks(self, agent_positions, agents_dones=None):
        """
        为每个智能体生成其 Voronoi 区域的二值掩码
    
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            agent_dones: 智能体完成状态列表 [bool, bool, ...] (可选)
    
        返回:
            masks: 列表，每个元素是一个 (map_dim, map_dim) 的布尔数组
                   masks[i] 表示第 i 个智能体的 Voronoi 区域掩码
                   区域内的栅格为 True，其他为 False
        """
        if len(agent_positions) == 0:
            return []
    
        # 计算 Voronoi 区域划分
        voronoi_map = self.compute_voronoi_regions(agent_positions, agents_dones)
    
        if voronoi_map is None:
            return []
    
        # 为每个智能体生成独立的掩码
        masks = []
        for agent_idx in range(len(agent_positions)):
            # 如果智能体已完成,返回空掩码
            if agents_dones is not None and agents_dones[agent_idx]:
                masks.append(np.zeros((self.map_dim, self.map_dim), dtype=bool))
            else:
                mask = (voronoi_map == agent_idx)
                masks.append(mask)
    
        return masks
    
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
    
    def detect_targets(self):
        """
        检测信念地图中的目标点
        returns:
            dict: {
                'binary_map': 二值化地图,
                'num_targets': 目标点数量,
                'target_positions': 目标点世界坐标列表,
                'target_grid_positions': 目标点栅格索引列表,
                'cluster_sizes': 每个目标点对应的簇大小列表,
                'cluster_labels': 簇标签地图
            }
        """
        binary_map = (self.belief_grid > self.belief_threshold).astype(np.int8)
        
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
    
    def get_target_positions(self, min_cluster_size=1):
        """直接返回检测到的目标点世界坐标"""
        result = self.detect_targets()
        
        if min_cluster_size > 1:
            filtered_positions = []
            for pos, size in zip(result['target_positions'], result['cluster_sizes']):
                if size >= min_cluster_size:
                    filtered_positions.append(pos)
            return filtered_positions
        
        return result['target_positions']
    
    def visualize_detected_targets(self):
        """可视化检测到的目标点"""
        result = self.detect_targets()
        
        vis_map = result['cluster_labels'].astype(np.float32)
        
        for grid_pos in result['target_grid_positions']:
            i, j = grid_pos
            vis_map[i, j] = -1
        
        result['visualization_map'] = vis_map
        return result
    
    def get_targets_summary(self):
        """获取目标检测的摘要信息"""
        result = self.detect_targets()
        
        summary = {
            'num_targets': result['num_targets'],
            'total_high_belief_cells': np.sum(result['binary_map']),
            'mean_cluster_size': np.mean(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'max_cluster_size': max(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'min_cluster_size': min(result['cluster_sizes']) if result['cluster_sizes'] else 0,
            'target_positions': result['target_positions']
        }
        
        return summary
    
    def get_agents_heatmap(self, agent_positions, radius, sigma=None, clip_outside=True):
        """
        生成表示所有智能体位置的组合热图
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            radius: 影响半径，半径外的值为0
            sigma: 高斯标准差，控制衰减速度。如果为None，则 sigma = radius / 3
            clip_outside: 是否在半径外截断为0 (True: 截断, False: 不截断)
        
        返回:
            heatmap: (map_dim, map_dim) 的浮点数组
                    每个智能体中心位置为1，在半径内高斯衰减，半径外为0
                    多个智能体的热图会叠加（取最大值）
        """
        if len(agent_positions) == 0:
            return np.zeros((self.map_dim, self.map_dim), dtype=np.float32)
        
        # 默认 sigma = radius / 3，这样在半径处约衰减到 0.01
        if sigma is None:
            sigma = radius / 3.0
        
        # 初始化热图
        heatmap = np.zeros((self.map_dim, self.map_dim), dtype=np.float32)
        
        for agent_pos in agent_positions:
            x, y = agent_pos
            
            # 计算每个栅格中心到智能体的距离
            dist_x = self.cell_world_x - x
            dist_y = self.cell_world_y - y
            distance = np.sqrt(dist_x**2 + dist_y**2)
            
            # 高斯函数: exp(-(d^2) / (2*sigma^2))
            agent_heatmap = np.exp(-(distance**2) / (2 * sigma**2))
            
            # 截断：半径外的值设为0
            if clip_outside:
                agent_heatmap[distance > radius] = 0.0
            
            # 叠加到总热图（取最大值，避免多个智能体重叠时值过大）
            heatmap = np.maximum(heatmap, agent_heatmap)
        
        return heatmap.astype(np.float32)
    
    def get_landmarks_heatmap(self, radius=None, sigma=None, clip_outside=True, landmark_positions=None):
        """
        生成表示所有 landmarks 位置的组合热图
        
        参数:
            radius: 影响半径，半径外的值为0。如果为None，使用 self.landmark_radius
            sigma: 高斯标准差，控制衰减速度。如果为None，则 sigma = radius / 3
            clip_outside: 是否在半径外截断为0 (True: 截断, False: 不截断)
            landmark_positions: landmarks 位置列表 [(x1, y1), (x2, y2), ...]
                            如果为None，使用 self.landmark_positions
        
        返回:
            heatmap: (map_dim, map_dim) 的浮点数组
                    每个 landmark 中心位置为1，在半径内高斯衰减，半径外为0
                    多个 landmarks 的热图会叠加（取最大值）
        """
        # 使用传入的 landmark_positions 或默认的 self.landmark_positions
        if landmark_positions is None:
            landmark_positions = self.landmark_positions
        
        if len(landmark_positions) == 0:
            return np.zeros((self.map_dim, self.map_dim), dtype=np.float32)
        
        # 使用默认的 landmark_radius 或传入的 radius
        if radius is None:
            radius = self.landmark_radius
        
        # 默认 sigma = radius / 3，这样在半径处约衰减到 0.01
        if sigma is None:
            sigma = radius / 3.0
        
        # 初始化热图
        heatmap = np.zeros((self.map_dim, self.map_dim), dtype=np.float32)
        
        for landmark_pos in landmark_positions:
            x, y = landmark_pos
            
            # 计算每个栅格中心到 landmark 的距离
            dist_x = self.cell_world_x - x
            dist_y = self.cell_world_y - y
            distance = np.sqrt(dist_x**2 + dist_y**2)
            
            # 高斯函数: exp(-(d^2) / (2*sigma^2))
            landmark_heatmap = np.exp(-(distance**2) / (2 * sigma**2))
            
            # 截断：半径外的值设为0
            if clip_outside:
                landmark_heatmap[distance > radius] = 0.0
            
            # 叠加到总热图（取最大值，避免多个 landmarks 重叠时值过大）
            heatmap = np.maximum(heatmap, landmark_heatmap)
        
        return heatmap.astype(np.float32)
    
    def get_agent_step_rewards(self, agent_positions, radius, discover_reward_scale=1.0):
        """
        计算每个智能体在其观测范围内的探索熵奖励和发现目标奖励
        
        探索奖励：r_explore^i = Entropy(M_{t-1}) - Entropy(M_{t-1} ∪ O_t^i)
        发现奖励：r_discover^i = sum_{c in FOV_i} max(0, b_c^new - b_c^old)
        
        参数:
            agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
            radius: 观测半径
            discover_reward_scale: 发现奖励的缩放系数
        
        返回:
            dict: {
                'explore_rewards': 列表，每个元素是对应智能体的探索奖励
                'discover_rewards': 列表，每个元素是对应智能体的发现目标奖励
            }
        """
        if len(agent_positions) == 0:
            return {'explore_rewards': [], 'discover_rewards': []}
        
        # 保存当前地图状态
        original_belief_grid = self.belief_grid.copy()
        
        # 计算当前地图的总熵 Entropy(M_{t-1})
        original_entropy_map = self.compute_shannon_entropy()
        original_total_entropy = np.sum(original_entropy_map)
        
        explore_rewards = []
        discover_rewards = []
        
        # 对每个智能体单独计算两种奖励
        for agent_pos in agent_positions:
            # 1. 恢复原始地图状态
            self.belief_grid = original_belief_grid.copy()
            
            # 2. 获取当前智能体的观测范围
            fov_mask = self.get_fov_mask(agent_pos, radius)
            
            # 正向更新：被观测到 且 包含 landmark 的区域
            positive_mask = fov_mask & self.landmark_map
            # 负向更新：被观测到 但 不包含 landmark 的区域
            negative_mask = fov_mask & (~self.landmark_map)
            
            # 3. 执行贝叶斯更新
            self.bayesian_update(positive_mask, negative_mask)
            
            # 4. 计算探索奖励（熵减少量）
            updated_entropy_map = self.compute_shannon_entropy()
            updated_total_entropy = np.sum(updated_entropy_map)
            explore_reward = original_total_entropy - updated_total_entropy
            explore_rewards.append(float(explore_reward / self.explore_reward_normalization) * 2)
            
            # 5. 计算发现目标奖励（信念正向增量）
            delta_belief = self.belief_grid - original_belief_grid
            positive_delta = np.maximum(0, delta_belief[fov_mask])
            total_discover = np.sum(positive_delta)
            discover_reward = total_discover * discover_reward_scale
            discover_rewards.append(float(discover_reward / self.discover_reward_normalization) * 5)
        
        # 恢复原始地图状态
        self.belief_grid = original_belief_grid
        
        return {
            'explore_rewards': explore_rewards,
            'discover_rewards': discover_rewards
        }

    def get_agent_step_explore_entropy(self, agent_positions, radius, sigma=None, clip_outside=True):
        """向后兼容的探索熵奖励接口"""
        return self.get_agent_step_rewards(agent_positions, radius)['explore_rewards']

    def get_agent_discover_target_reward(self, agent_positions, radius, reward_value=1.0):
        """向后兼容的发现目标奖励接口"""
        return self.get_agent_step_rewards(agent_positions, radius, reward_value)['discover_rewards']