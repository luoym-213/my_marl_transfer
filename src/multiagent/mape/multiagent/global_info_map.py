"""
Global Information Map for tracking exploration progress in multi-agent environments.
"""
import numpy as np

class GlobalInfoMap:
    """
    离散化的全局信息地图，用于跟踪团队的探索进度。
    
    参数:
        world_size: 世界的总尺寸 (从 -world_size/2 到 +world_size/2)
        cell_size: 每个栅格的尺寸
    """
    
    def __init__(self, world_size=2.0, cell_size=0.05):
        self.world_size = world_size
        self.cell_size = cell_size
        
        # 计算地图维度: 2 / 0.05 = 40
        self.map_dim = int(world_size / cell_size)
        
        # 初始化栅格地图 (0=未探索, 1=已探索)
        self.grid = np.zeros((self.map_dim, self.map_dim), dtype=np.int8)
        
        # 世界坐标的边界(-1,1)
        self.world_min = -world_size / 2.0
        self.world_max = world_size / 2.0
        
        # 预计算每个栅格中心的世界坐标 (用于高效距离计算)
        self._precompute_cell_centers()
        
    def _precompute_cell_centers(self):
        """预计算每个栅格中心点的世界坐标把离散网格坐标映射到连续世界坐标，
        为后续的距离计算（如视野范围、探索范围判断）提供高效的预计算坐标表。"""
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
        """重置地图，将所有栅格标记为未探索"""
        self.grid.fill(0)
        
    def world_to_grid(self, world_pos):
        """
        将世界坐标转换为栅格索引
        
        参数:
            world_pos: (x, y) 世界坐标
            
        返回:
            (i, j) 栅格索引，如果超出边界则返回 None
        将世界坐标系中的连续位置 (x, y)转换为离散地图中的格子索引 (i, j)，
        用于在全局信息地图上定位、更新或查询某个位置的探索状态。
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
    
    def update_explored_area(self, mask):
        """
        更新地图的探索状态
        
        参数:
            mask: (map_dim, map_dim) 布尔数组，True 表示新探索的区域
        """
        self.grid[mask] = 1
        
    def get_exploration_rate(self):
        """
        计算当前的探索率
        
        返回:
            探索率 (0.0 到 1.0)
        """
        return np.sum(self.grid) / (self.map_dim * self.map_dim)
    
    def get_unexplored_count(self):
        """返回未探索的栅格数量"""
        return np.sum(self.grid == 0)
    
    def visualize(self):
        """
        返回可视化的地图副本 (用于调试)
        
        返回:
            (map_dim, map_dim) 的数组
        """
        return self.grid.copy()