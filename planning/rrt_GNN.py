"""

RRT for entropy-based exploration with Voronoi mask sampling

基于熵值和Voronoi区域的RRT探索算法

"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Sequence, Optional, Any

from scipy.ndimage import convolve  # 新增：用于快速计算半径内累积熵


# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class RRT_GNN:
    """
    基于熵值探索的RRT类
    从Voronoi区域内高熵值区域采样，计算节点价值并返回Top-K节点
    """

    class Node:
        """
        RRT节点，包含位置、价值和父节点信息
        """

        def __init__(self, x: int, y: int):
            self.x = x  # 栅格索引x
            self.y = y  # 栅格索引y
            self.value = 0.0  # 节点价值（累计熵值）
            self.parent = None  # 父节点

        def __repr__(self):
            return f"Node(x={self.x}, y={self.y}, value={self.value:.2f})"

    def __init__(self,
                 start: List[int],
                 voronoi_mask: np.ndarray,
                 entropy_map: np.ndarray,
                 entropy_threshold: float = 0.5,
                 radius: int = 2,
                 expand_dis: int = 2,
                 max_iterations: int = 500,
                 top_k: int = 10):
        """
        初始化RRT_GNN

        参数:
            start: 起始位置 [x, y]，智能体当前位置（栅格索引）
            voronoi_mask: [100, 100] 布尔数组，智能体的Voronoi区域掩码
            entropy_map: [100, 100] 数组，每个栅格的香农熵值
            entropy_threshold: 熵值阈值，只在熵值>此阈值的区域采样，默认0.8
            radius: 计算节点价值时的半径（栅格单位）
            expand_dis: RRT扩展距离/生长步长（栅格单位）
            max_iterations: 最大迭代次数
            top_k: 返回价值最高的K个节点
        """
        self.start = self.Node(start[0], start[1])
        self.voronoi_mask = voronoi_mask
        self.entropy_map = entropy_map
        self.entropy_threshold = entropy_threshold
        self.radius = int(radius)
        self.expand_dis = expand_dis
        self.max_iterations = max_iterations
        self.top_k = top_k
        
        # 生成采样列表：Voronoi区域内高熵值区域的索引
        self.sample_list = self._generate_sample_list()
        
        # 节点列表
        self.node_list = []
        
        # 地图尺寸
        self.map_size = entropy_map.shape

        # 新增：预计算“半径内熵累计和”整图（O(H*W*r^2) 一次，后面查询 O(1)）
        self._disk_kernel = self._make_disk_kernel(self.radius)
        self.local_entropy_map = convolve(
            self.entropy_map.astype(np.float32),
            self._disk_kernel,
            mode="constant",
            cval=0.0,
        )

    @staticmethod
    def _make_disk_kernel(radius: int) -> np.ndarray:
        """生成半径为 radius 的圆盘核（0/1），用于累计半径内像素和。"""
        if radius <= 0:
            return np.array([[1.0]], dtype=np.float32)

        d = 2 * radius + 1
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = (xx * xx + yy * yy) <= (radius * radius)
        kernel = mask.astype(np.float32)
        return kernel

    def _generate_sample_list(self) -> List[Tuple[int, int]]:
        """
        生成采样列表：Voronoi区域内香农熵大于阈值的栅格索引 + Voronoi区域内随机100个索引

        返回:
            sample_list: [(x1, y1), (x2, y2), ...] 索引列表
        """
        # 找到满足条件的索引：在Voronoi区域内且熵值大于阈值
        high_entropy_mask = (self.voronoi_mask) & (self.entropy_map > self.entropy_threshold)
        high_entropy_indices = np.argwhere(high_entropy_mask)
        
        # 转换为列表 [(x, y), ...]
        sample_list = [(int(idx[0]), int(idx[1])) for idx in high_entropy_indices]
        
        # 额外在Voronoi区域内随机选择100个索引
        voronoi_indices = np.argwhere(self.voronoi_mask)
        if len(voronoi_indices) > 0:
            # 随机选择最多100个索引（如果区域小于100，就全选）
            n_random = min(100, len(voronoi_indices))
            random_idx = np.random.choice(len(voronoi_indices), size=n_random, replace=False)
            random_samples = [(int(voronoi_indices[i][0]), int(voronoi_indices[i][1])) 
                            for i in random_idx]
            
            # 合并两部分（去重）
            sample_list = list(set(sample_list + random_samples))
        
        return sample_list

    def _calculate_local_entropy(self, x: int, y: int) -> float:
        """O(1) 查询：直接查表，而不是每次遍历半径邻域。"""
        return float(self.local_entropy_map[x, y])

    def _get_random_node(self) -> 'RRT_GNN.Node':
        """
        从sample_list中随机采样一个节点

        返回:
            rnd_node: 随机采样的节点
        """
        if len(self.sample_list) == 0:
            raise ValueError("sample_list为空，无法采样！请检查voronoi_mask和entropy_map")
        
        # 从sample_list中随机选择一个位置
        x, y = random.choice(self.sample_list)
        rnd_node = self.Node(x, y)
        
        return rnd_node

    def _get_nearest_node_index(self, node_list: List['RRT_GNN.Node'], 
                                 rnd_node: 'RRT_GNN.Node') -> int:
        """
        找到node_list中距离rnd_node最近的节点索引

        参数:
            node_list: 节点列表
            rnd_node: 随机采样的节点

        返回:
            minind: 最近节点的索引
        """
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        
        return minind

    def _steer(self, from_node: 'RRT_GNN.Node', 
               to_node: 'RRT_GNN.Node') -> 'RRT_GNN.Node':
        """
        从from_node向to_node方向扩展expand_dis距离，生成新节点

        参数:
            from_node: 起始节点
            to_node: 目标节点

        返回:
            new_node: 扩展后的新节点
        """
        # 计算方向
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.hypot(dx, dy)
        
        if dist == 0:
            return None
        
        # 计算新节点位置
        if dist <= self.expand_dis:
            # 如果距离小于扩展距离，直接到达目标点
            new_x = to_node.x
            new_y = to_node.y
        else:
            # 否则按扩展距离移动
            new_x = int(from_node.x + self.expand_dis * dx / dist)
            new_y = int(from_node.y + self.expand_dis * dy / dist)
        
        # 检查新节点是否在地图范围内
        if not (0 <= new_x < self.map_size[0] and 0 <= new_y < self.map_size[1]):
            return None
        
        # 创建新节点
        new_node = self.Node(new_x, new_y)
        new_node.parent = from_node
        
        # 计算新节点的价值：自身周围熵值 + 父节点价值
        local_entropy = self._calculate_local_entropy(new_x, new_y)
        new_node.value = local_entropy + from_node.value
        
        return new_node

    def planning(self) -> List[List[float]]:
        """
        执行RRT规划，返回价值最高的K个节点

        返回:
            top_k_nodes: 包含K个节点信息的列表，每个元素为 [x, y, value]
        """
        # 初始化起始节点的价值
        self.start.value = self._calculate_local_entropy(self.start.x, self.start.y)
        self.node_list = [self.start]
        
        # 迭代扩展
        for i in range(self.max_iterations):
            # 随机采样一个节点
            rnd_node = self._get_random_node()
            
            # 找到最近的节点
            nearest_ind = self._get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            # 向采样点扩展
            new_node = self._steer(nearest_node, rnd_node)
            
            # 如果新节点有效，添加到节点列表
            if new_node is not None:
                self.node_list.append(new_node)
        
        # 按价值排序，选择Top-K节点
        sorted_nodes = sorted(self.node_list, key=lambda node: node.value, reverse=True)
        top_k_nodes = sorted_nodes[:min(self.top_k, len(sorted_nodes))]
        
        # 格式化返回结果，value进行归一化  / 1500
        result = [
            [node.x, node.y, node.value / 1500]
            for node in top_k_nodes
        ]
        
        return result


def compute_voronoi_regions(agent_positions: List[Tuple[int, int]], 
                           map_size: Tuple[int, int] = (100, 100)) -> List[np.ndarray]:
    """
    计算多个智能体的Voronoi区域划分
    
    参数:
        agent_positions: 智能体位置列表 [(x1, y1), (x2, y2), ...]
        map_size: 地图大小 (width, height)
    
    返回:
        voronoi_masks: 每个智能体的Voronoi区域掩码列表
    """
    n_agents = len(agent_positions)
    voronoi_masks = []
    
    # 创建网格
    x_grid, y_grid = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]), indexing='ij')
    
    # 对每个栅格，计算它属于哪个智能体（最近的智能体）
    for agent_idx in range(n_agents):
        # 计算当前智能体到所有栅格的距离
        agent_x, agent_y = agent_positions[agent_idx]
        dist_to_agent = (x_grid - agent_x)**2 + (y_grid - agent_y)**2
        
        # 初始化掩码
        mask = np.ones(map_size, dtype=bool)
        
        # 检查是否是最近的智能体
        for other_idx in range(n_agents):
            if other_idx != agent_idx:
                other_x, other_y = agent_positions[other_idx]
                dist_to_other = (x_grid - other_x)**2 + (y_grid - other_y)**2
                # 如果其他智能体更近，则不属于当前智能体的区域
                mask = mask & (dist_to_agent <= dist_to_other)
        
        voronoi_masks.append(mask)
    
    return voronoi_masks


def main():
    """示例代码，包含多智能体Voronoi划分和可视化"""
    print("RRT_GNN 多智能体示例 - 基于熵值的探索")
    
    # 创建示例数据
    np.random.seed(42)  # 设置随机种子以便复现
    
    # 随机生成3个智能体的位置
    n_agents = 3
    agent_positions = [
        (25, 30),  # 智能体1
        (70, 50),  # 智能体2
        (45, 75),  # 智能体3
    ]
    agent_colors = ['green', 'blue', 'purple']  # 每个智能体的颜色
    
    print(f"\n智能体位置:")
    for i, pos in enumerate(agent_positions):
        print(f"  智能体{i+1}: {pos}")
    
    # 计算Voronoi区域划分
    print("\n计算Voronoi区域划分...")
    voronoi_masks = compute_voronoi_regions(agent_positions, map_size=(100, 100))
    
    for i, mask in enumerate(voronoi_masks):
        print(f"  智能体{i+1}的Voronoi区域大小: {np.sum(mask)} 个栅格")
    
    # 创建熵值分布（多个高斯峰）
    x_grid, y_grid = np.meshgrid(np.arange(100), np.arange(100), indexing='ij')
    entropy_map = np.zeros((100, 100))
    
    # 添加多个高熵值区域
    peaks = [(30, 40, 0.9), (60, 50, 0.95), (45, 65, 0.85), 
             (20, 20, 0.88), (75, 75, 0.92), (80, 30, 0.87)]
    for px, py, peak_val in peaks:
        gaussian = np.exp(-((x_grid - px)**2 + (y_grid - py)**2) / (2 * 10**2))
        entropy_map += peak_val * gaussian
    
    # 添加一些噪声
    entropy_map += np.random.rand(100, 100) * 0.3
    entropy_map = np.clip(entropy_map, 0, 1)  # 限制在[0,1]范围
    
    # 为每个智能体运行RRT_GNN
    all_results = []
    for i, (agent_pos, voronoi_mask) in enumerate(zip(agent_positions, voronoi_masks)):
        print(f"\n=== 智能体{i+1} 执行RRT规划 ===")
        
        # 初始化RRT_GNN
        rrt = RRT_GNN(
            start=list(agent_pos),
            voronoi_mask=voronoi_mask,
            entropy_map=entropy_map,
            entropy_threshold=0.8,
            radius=5,
            expand_dis=3,
            max_iterations=300,
            top_k=5  # 每个智能体返回5个候选点
        )
        
        # 执行规划
        print(f"  采样列表大小: {len(rrt.sample_list)}")
        top_k_nodes = rrt.planning()
        
        # 打印结果
        print(f"  找到 {len(top_k_nodes)} 个高价值节点:")
        for j, node_info in enumerate(top_k_nodes):
            print(f"    {j+1}. 位置: ({node_info[0]}, {node_info[1]}), 价值: {node_info[2]:.2f}")
        
        all_results.append({
            'agent_id': i,
            'position': agent_pos,
            'color': agent_colors[i],
            'voronoi_mask': voronoi_mask,
            'top_k_nodes': top_k_nodes,
            'rrt': rrt
        })
    
    # ===== 可视化 =====
    print("\n生成可视化...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 1. 绘制香农熵热力图
    im = ax.imshow(entropy_map.T, cmap='hot', origin='lower', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Shannon Entropy', shrink=0.8)
    
    # 2. 绘制每个智能体的Voronoi区域边界
    for result in all_results:
        voronoi_mask = result['voronoi_mask']
        color = result['color']
        agent_id = result['agent_id']
        
        # 使用contour在边界处绘制线条
        contour = ax.contour(voronoi_mask.T, levels=[0.5], colors=color, 
                           linewidths=2.5, origin='lower', alpha=0.8)
        
        # 添加区域标签
        # 找到区域中心
        indices = np.argwhere(voronoi_mask)
        if len(indices) > 0:
            center = np.mean(indices, axis=0)
            ax.text(center[0], center[1], f'Agent {agent_id+1}', 
                   fontsize=12, fontweight='bold', color=color,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=color, linewidth=2))
    
    # 3. 绘制每个智能体的RRT树和候选点
    for result in all_results:
        agent_id = result['agent_id']
        agent_pos = result['position']
        color = result['color']
        top_k_nodes = result['top_k_nodes']
        rrt = result['rrt']
        
        # 标记智能体起始位置
        ax.plot(agent_pos[0], agent_pos[1], '*', color=color, markersize=25, 
               markeredgecolor='white', markeredgewidth=2, label=f'Agent {agent_id+1} Start',
               zorder=10)
        
        # 绘制RRT树结构（半透明）
        for node in rrt.node_list:
            if node.parent is not None:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                       color=color, alpha=0.15, linewidth=0.5, zorder=1)
        
        # 标记Top-K候选点
        for j, node_info in enumerate(top_k_nodes):
            pos = (node_info[0], node_info[1])
            val = node_info[2]
            if j == 0:  # 最高价值的点
                ax.plot(pos[0], pos[1], 'D', color=color, markersize=12, 
                       markeredgecolor='yellow', markeredgewidth=2, zorder=8)
                ax.text(pos[0]+2, pos[1]+2, f'#{j+1}\nv={val:.1f}', 
                       color='white', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                       zorder=9)
            else:
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=1, zorder=8)
                ax.text(pos[0]+2, pos[1]+2, f'#{j+1}', 
                       color='white', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                       zorder=9)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=15, label='智能体起始位置', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
               markersize=12, label='最高价值候选点', markeredgecolor='yellow', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='其他候选点', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], color='gray', linewidth=2.5, label='Voronoi边界'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('X (grid index)', fontsize=12)
    ax.set_ylabel('Y (grid index)', fontsize=12)
    ax.set_title('多智能体RRT_GNN: 基于Voronoi划分的探索规划', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()


def plan_batch(
    starts: Sequence[Sequence[int]],
    voronoi_masks: np.ndarray,
    entropy_maps: np.ndarray,
    entropy_threshold: float = 0.8,
    radius: int = 5,
    expand_dis: int = 3,
    max_iterations: int = 500,
    top_k: int = 10,
) -> List[List[List[float]]]:
    """
    Batch 版本：对 B 份数据逐个运行 RRT_GNN（不是向量化并行，只是统一入口）

    输入:
      - starts: (B, 2) 的栅格索引 [[x,y], ...]
      - voronoi_masks: (B, H, W) bool
      - entropy_maps: (B, H, W) float

    输出:
      - results: 长度为 B 的 list；每个元素是 RRT_GNN.planning() 的返回(list[list])
                 即 [[x, y, value], ...]
    """
    if voronoi_masks.ndim != 3 or entropy_maps.ndim != 3:
        raise ValueError("voronoi_masks 与 entropy_maps 需要是 (B,H,W)")
    if len(starts) != voronoi_masks.shape[0] or len(starts) != entropy_maps.shape[0]:
        raise ValueError("starts 与 masks/maps 的 batch 维度 B 不一致")

    results: List[List[dict]] = []
    B = voronoi_masks.shape[0]
    for b in range(B):
        rrt = RRT_GNN(
            start=list(starts[b]),
            voronoi_mask=voronoi_masks[b],
            entropy_map=entropy_maps[b],
            entropy_threshold=entropy_threshold,
            radius=radius,
            expand_dis=expand_dis,
            max_iterations=max_iterations,
            top_k=top_k,
        )
        results.append(rrt.planning())
    return results


if __name__ == '__main__':
    main()
