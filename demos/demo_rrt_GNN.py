"""
RRT_GNN 外部调用示例
演示如何从外部文件调用RRT_GNN进行多智能体探索规划
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加planning目录到路径
sys.path.append('/Users/ym/Public/codes/marl_transfer/planning')
from rrt_GNN import RRT_GNN, compute_voronoi_regions

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """RRT_GNN外部调用示例 - 多智能体场景"""
    print("=" * 60)
    print("RRT_GNN 外部调用示例 - 多智能体探索规划")
    print("=" * 60)
    
    # ===== 1. 准备数据 =====
    np.random.seed(42)  # 设置随机种子以便复现
    
    # 智能体位置
    n_agents = 3
    agent_positions = [
        (25, 30),  # 智能体1
        (70, 50),  # 智能体2
        (45, 75),  # 智能体3
    ]
    agent_colors = ['green', 'blue', 'purple']
    
    print(f"\n【步骤1】智能体初始化")
    print(f"智能体数量: {n_agents}")
    for i, pos in enumerate(agent_positions):
        print(f"  智能体{i+1}位置: {pos}")
    
    # ===== 2. 计算Voronoi区域划分 =====
    print(f"\n【步骤2】计算Voronoi区域划分")
    voronoi_masks = compute_voronoi_regions(agent_positions, map_size=(100, 100))
    
    for i, mask in enumerate(voronoi_masks):
        print(f"  智能体{i+1}的Voronoi区域: {np.sum(mask)} 个栅格")
    
    # ===== 3. 创建熵值地图 =====
    print(f"\n【步骤3】生成熵值地图")
    x_grid, y_grid = np.meshgrid(np.arange(100), np.arange(100), indexing='ij')
    entropy_map = np.zeros((100, 100))
    
    # 添加多个高熵值区域（模拟未探索的高不确定性区域）
    peaks = [(30, 40, 0.9), (60, 50, 0.95), (45, 65, 0.85), 
             (20, 20, 0.88), (75, 75, 0.92), (80, 30, 0.87)]
    print(f"  高熵值峰值数量: {len(peaks)}")
    for px, py, peak_val in peaks:
        gaussian = np.exp(-((x_grid - px)**2 + (y_grid - py)**2) / (2 * 10**2))
        entropy_map += peak_val * gaussian
    
    # 添加噪声
    entropy_map += np.random.rand(100, 100) * 0.3
    entropy_map = np.clip(entropy_map, 0, 1)
    print(f"  熵值范围: [{entropy_map.min():.2f}, {entropy_map.max():.2f}]")
    
    # ===== 4. 为每个智能体执行RRT规划 =====
    print(f"\n【步骤4】执行RRT规划")
    all_results = []
    
    for i, (agent_pos, voronoi_mask) in enumerate(zip(agent_positions, voronoi_masks)):
        print(f"\n--- 智能体{i+1} RRT规划 ---")
        
        # 初始化RRT_GNN
        rrt = RRT_GNN(
            start=list(agent_pos),
            voronoi_mask=voronoi_mask,
            entropy_map=entropy_map,
            entropy_threshold=0.8,
            radius=5,
            expand_dis=3,
            max_iterations=300,
            top_k=5
        )
        
        print(f"  采样区域大小: {len(rrt.sample_list)} 个栅格")
        
        # 执行规划
        top_k_nodes = rrt.planning()
        
        print(f"  找到 {len(top_k_nodes)} 个候选点:")
        for j, node_info in enumerate(top_k_nodes):
            print(f"    #{j+1}: 位置{node_info['position']}, 价值{node_info['value']:.2f}")
        
        # 保存结果
        all_results.append({
            'agent_id': i,
            'position': agent_pos,
            'color': agent_colors[i],
            'voronoi_mask': voronoi_mask,
            'top_k_nodes': top_k_nodes,
            'rrt': rrt
        })
    
    # ===== 5. 结果分析 =====
    print(f"\n【步骤5】结果分析")
    print(f"\n总体统计:")
    total_nodes = sum(len(r['rrt'].node_list) for r in all_results)
    print(f"  总生成节点数: {total_nodes}")
    for i, result in enumerate(all_results):
        best_value = result['top_k_nodes'][0]['value']
        best_pos = result['top_k_nodes'][0]['position']
        print(f"  智能体{i+1}最优目标: {best_pos}, 价值: {best_value:.2f}")
    
    # ===== 6. 可视化 =====
    print(f"\n【步骤6】生成可视化")
    visualize_results(entropy_map, all_results)
    
    print(f"\n{'='*60}")
    print("示例运行完成！")
    print(f"{'='*60}")


def visualize_results(entropy_map, all_results):
    """可视化结果"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 1. 绘制香农熵热力图
    im = ax.imshow(entropy_map.T, cmap='hot', origin='lower', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Shannon Entropy', shrink=0.8)
    
    # 2. 绘制每个智能体的Voronoi区域边界
    for result in all_results:
        voronoi_mask = result['voronoi_mask']
        color = result['color']
        agent_id = result['agent_id']
        
        # Voronoi边界
        ax.contour(voronoi_mask.T, levels=[0.5], colors=color, 
                  linewidths=2.5, origin='lower', alpha=0.8)
        
        # 区域标签
        indices = np.argwhere(voronoi_mask)
        if len(indices) > 0:
            center = np.mean(indices, axis=0)
            ax.text(center[0], center[1], f'Agent {agent_id+1}', 
                   fontsize=12, fontweight='bold', color=color,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, 
                           edgecolor=color, linewidth=2))
    
    # 3. 绘制智能体和候选点
    for result in all_results:
        agent_id = result['agent_id']
        agent_pos = result['position']
        color = result['color']
        top_k_nodes = result['top_k_nodes']
        rrt = result['rrt']
        
        # 起始位置
        ax.plot(agent_pos[0], agent_pos[1], '*', color=color, markersize=25, 
               markeredgecolor='white', markeredgewidth=2, zorder=10)
        
        # RRT树结构
        for node in rrt.node_list:
            if node.parent is not None:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                       color=color, alpha=0.15, linewidth=0.5, zorder=1)
        
        # 候选点
        for j, node_info in enumerate(top_k_nodes):
            pos = node_info['position']
            if j == 0:  # 最高价值点
                ax.plot(pos[0], pos[1], 'D', color=color, markersize=12, 
                       markeredgecolor='yellow', markeredgewidth=2, zorder=8)
                ax.text(pos[0]+2, pos[1]+2, f'#{j+1}\nv={node_info["value"]:.1f}', 
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
    
    # 图例
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
    ax.set_title('RRT_GNN外部调用示例：多智能体探索规划', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
