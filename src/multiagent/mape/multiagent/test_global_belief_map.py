"""
测试 GlobalBeliefMap 的各项功能
"""
import numpy as np
import matplotlib.pyplot as plt
from global_belief_map import GlobalBeliefMap

def test_basic_initialization():
    """测试基本初始化"""
    print("=" * 50)
    print("测试1: 基本初始化")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap(world_size=2.0, cell_size=0.02)
    
    print(f"地图维度: {belief_map.map_dim} x {belief_map.map_dim}")
    print(f"世界范围: [{belief_map.world_min}, {belief_map.world_max}]")
    print(f"初始信念值: {belief_map.initial_belief}")
    print(f"传感器保真度: {belief_map.sensor_fidelity}")
    print(f"信念地图形状: {belief_map.belief_grid.shape}")
    print(f"初始平均信念: {np.mean(belief_map.belief_grid):.3f}")
    print(f"初始总不确定性: {belief_map.get_total_uncertainty():.2f}")
    print(f"初始平均不确定性: {belief_map.get_mean_uncertainty():.4f}")
    print()

def test_coordinate_conversion():
    """测试坐标转换"""
    print("=" * 50)
    print("测试2: 坐标转换")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 测试世界坐标到栅格索引
    test_positions = [
        (0.0, 0.0),      # 中心
        (0.5, 0.5),      # 右上
        (-0.5, -0.5),    # 左下
        (0.99, 0.99),    # 边界
        (-1.0, -1.0),    # 边界
    ]
    
    print("世界坐标 -> 栅格索引 -> 世界坐标")
    for pos in test_positions:
        grid_pos = belief_map.world_to_grid(pos)
        if grid_pos:
            world_pos = belief_map.grid_to_world(grid_pos)
            print(f"{pos} -> {grid_pos} -> {world_pos}")
        else:
            print(f"{pos} -> 超出边界")
    print()

def test_bayesian_update():
    """测试贝叶斯更新"""
    print("=" * 50)
    print("测试3: 贝叶斯信念更新")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 智能体位置
    agent_pos = (0.0, 0.0)
    obs_radius = 0.3
    
    print(f"智能体位置: {agent_pos}")
    print(f"观测半径: {obs_radius}")
    print(f"更新前中心点信念: {belief_map.get_belief_at_position(agent_pos):.4f}")
    
    # 进行5次更新
    for i in range(5):
        belief_map.update_beliefs([agent_pos], obs_radius)
        belief = belief_map.get_belief_at_position(agent_pos)
        uncertainty = belief_map.get_mean_uncertainty()
        print(f"更新 {i+1} 次后 - 中心点信念: {belief:.4f}, 平均不确定性: {uncertainty:.4f}")
    print()

def test_multiple_agents():
    """测试多智能体场景"""
    print("=" * 50)
    print("测试4: 多智能体信念更新")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 3个智能体
    agent_positions = [
        (0.3, 0.3),
        (-0.3, -0.3),
        (0.0, 0.5)
    ]
    obs_radius = 0.2
    
    print(f"智能体数量: {len(agent_positions)}")
    print(f"智能体位置: {agent_positions}")
    print(f"观测半径: {obs_radius}")
    
    print(f"\n更新前总不确定性: {belief_map.get_total_uncertainty():.2f}")
    
    # 更新3次
    for i in range(3):
        belief_map.update_beliefs(agent_positions, obs_radius)
        total_unc = belief_map.get_total_uncertainty()
        mean_unc = belief_map.get_mean_uncertainty()
        print(f"更新 {i+1} 次后 - 总不确定性: {total_unc:.2f}, 平均不确定性: {mean_unc:.4f}")
    print()

def test_voronoi_and_centroids():
    """测试Voronoi划分和熵加权质心"""
    print("=" * 50)
    print("测试5: Voronoi划分和熵加权质心")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 3个智能体
    agent_positions = [
        (0.4, 0.4),
        (-0.4, -0.4),
        (0.0, 0.0)
    ]
    
    print(f"智能体位置: {agent_positions}")
    
    # 先进行一些观测，创建不均匀的信念分布
    obs_radius = 0.3
    belief_map.update_beliefs([agent_positions[0]], obs_radius)
    belief_map.update_beliefs([agent_positions[1]], obs_radius)
    
    # 计算Voronoi区域
    voronoi_map = belief_map.compute_voronoi_regions(agent_positions)
    print(f"\nVoronoi地图形状: {voronoi_map.shape}")
    
    # 计算熵加权质心
    centroids = belief_map.compute_entropy_weighted_centroids(agent_positions)
    print("\n熵加权质心:")
    for i, (agent_pos, centroid) in enumerate(zip(agent_positions, centroids)):
        distance = np.sqrt((centroid[0] - agent_pos[0])**2 + (centroid[1] - agent_pos[1])**2)
        print(f"  智能体 {i}: 位置 {agent_pos} -> 质心 {centroid}")
        print(f"           距离: {distance:.4f}")
    
    # 获取区域统计信息
    stats = belief_map.get_voronoi_region_stats(agent_positions)
    print("\nVoronoi区域统计:")
    for stat in stats:
        print(f"  智能体 {stat['agent_idx']}:")
        print(f"    区域面积: {stat['area']} 栅格")
        print(f"    总熵: {stat['total_entropy']:.2f}")
        print(f"    平均熵: {stat['mean_entropy']:.4f}")
        print(f"    平均信念: {stat['mean_belief']:.4f}")
    print()

def test_high_uncertainty_regions():
    """测试高不确定性区域检测"""
    print("=" * 50)
    print("测试6: 高不确定性区域检测")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 在部分区域进行观测
    agent_positions = [(0.3, 0.3)]
    obs_radius = 0.4
    
    # 多次更新降低部分区域的不确定性
    for _ in range(5):
        belief_map.update_beliefs(agent_positions, obs_radius)
    
    # 检测高不确定性区域
    high_unc_positions = belief_map.get_high_uncertainty_positions(threshold=0.95)
    
    print(f"高不确定性栅格数量 (阈值=0.95): {len(high_unc_positions)}")
    print(f"总栅格数量: {belief_map.map_dim ** 2}")
    print(f"高不确定性区域占比: {len(high_unc_positions) / (belief_map.map_dim ** 2) * 100:.2f}%")
    
    # 显示几个高不确定性位置的世界坐标
    if len(high_unc_positions) > 0:
        print("\n前5个高不确定性位置的世界坐标:")
        for i, grid_pos in enumerate(high_unc_positions[:5]):
            world_pos = belief_map.grid_to_world(grid_pos)
            entropy = belief_map.compute_shannon_entropy()[grid_pos]
            print(f"  栅格 {grid_pos} -> 世界坐标 {world_pos}, 熵: {entropy:.4f}")
    print()

def visualize_belief_map():
    """可视化信念地图"""
    print("=" * 50)
    print("测试7: 可视化信念地图和熵地图")
    print("=" * 50)
    
    belief_map = GlobalBeliefMap()
    
    # 设置3个智能体
    agent_positions = [
        (0.5, 0.5),
        (-0.5, -0.5),
        (0.0, 0.6)
    ]
    obs_radius = 0.3
    
    # 进行多次观测
    for _ in range(5):
        belief_map.update_beliefs(agent_positions, obs_radius)
    
    # 计算Voronoi和质心
    voronoi_map = belief_map.visualize_voronoi(agent_positions)
    centroids = belief_map.compute_entropy_weighted_centroids(agent_positions)
    entropy_map = belief_map.visualize_entropy()
    belief_values = belief_map.visualize_beliefs()
    
    # 输出熵的统计信息
    print(f"\n熵分布统计:")
    print(f"  最小熵: {np.min(entropy_map):.4f}")
    print(f"  最大熵: {np.max(entropy_map):.4f}")
    print(f"  平均熵: {np.mean(entropy_map):.4f}")
    print(f"  总不确定性: {belief_map.get_total_uncertainty():.2f}")
    
    print(f"\n信念分布统计:")
    print(f"  最小信念: {np.min(belief_values):.4f}")
    print(f"  最大信念: {np.max(belief_values):.4f}")
    print(f"  平均信念: {np.mean(belief_values):.4f}")
    
    # 验证质心是否偏向高熵区域
    print(f"\n质心熵加权验证:")
    for i, (pos, centroid) in enumerate(zip(agent_positions, centroids)):
        agent_entropy = belief_map.compute_shannon_entropy()[belief_map.world_to_grid(pos)]
        centroid_grid = belief_map.world_to_grid(centroid)
        if centroid_grid:
            centroid_entropy = belief_map.compute_shannon_entropy()[centroid_grid]
            print(f"  智能体{i}: 位置处熵={agent_entropy:.4f}, 质心处熵={centroid_entropy:.4f}")
            print(f"           → 质心{'偏向高熵区域 ✓' if centroid_entropy > agent_entropy else '未偏向高熵区域'}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 信念地图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(belief_values.T, 
                     origin='lower', cmap='viridis', 
                     extent=[belief_map.world_min, belief_map.world_max,
                            belief_map.world_min, belief_map.world_max],
                     vmin=0, vmax=1)
    ax1.set_title('信念地图 (Belief Map)\n高信念=已探索', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Belief Value')
    cbar1.ax.text(0.5, -0.1, f'Min: {np.min(belief_values):.3f}\nMax: {np.max(belief_values):.3f}', 
                  transform=cbar1.ax.transAxes, ha='center', fontsize=8)
    
    # 标记智能体位置
    for i, pos in enumerate(agent_positions):
        ax1.plot(pos[0], pos[1], 'r*', markersize=20, markeredgecolor='white', 
                markeredgewidth=1.5, label=f'Agent {i}' if i == 0 else '')
        # 画观测范围圆
        circle = plt.Circle(pos, obs_radius, color='red', fill=False, 
                           linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.add_patch(circle)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 熵地图（不确定性分布）
    ax2 = axes[0, 1]
    # 使用实际的最小最大值来设置颜色范围，以显示熵的差异
    entropy_min, entropy_max = np.min(entropy_map), np.max(entropy_map)
    im2 = ax2.imshow(entropy_map.T, 
                     origin='lower', cmap='RdYlGn_r',  # 红黄绿反转：红=高熵，绿=低熵
                     extent=[belief_map.world_min, belief_map.world_max,
                            belief_map.world_min, belief_map.world_max],
                     vmin=entropy_min, vmax=entropy_max)  # 使用实际范围
    ax2.set_title(f'香农熵地图 (Entropy/Uncertainty Map)\n红色=高不确定性, 绿色=低不确定性 | 范围:[{entropy_min:.3f}, {entropy_max:.3f}]', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Shannon Entropy (bits)')
    cbar2.ax.text(0.5, -0.1, f'Min: {entropy_min:.4f}\nMax: {entropy_max:.4f}\nMean: {np.mean(entropy_map):.4f}', 
                  transform=cbar2.ax.transAxes, ha='center', fontsize=8)
    
    # 标记智能体位置
    for i, pos in enumerate(agent_positions):
        ax2.plot(pos[0], pos[1], 'cyan', marker='*', markersize=20, 
                markeredgecolor='white', markeredgewidth=1.5)
        ax2.text(pos[0], pos[1] - 0.08, f'A{i}', ha='center', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    ax2.grid(True, alpha=0.3)
    
    # 3. Voronoi划分 + 质心
    ax3 = axes[1, 0]
    im3 = ax3.imshow(voronoi_map.T, 
                     origin='lower', cmap='tab10', alpha=0.5,
                     extent=[belief_map.world_min, belief_map.world_max,
                            belief_map.world_min, belief_map.world_max])
    ax3.set_title('Voronoi区域划分 + 熵加权质心\n箭头指向质心', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    cbar3 = plt.colorbar(im3, ax=ax3, label='Agent ID', ticks=range(len(agent_positions)))
    
    # 标记智能体位置和质心
    for i, (pos, centroid) in enumerate(zip(agent_positions, centroids)):
        # 智能体位置
        ax3.plot(pos[0], pos[1], 'r*', markersize=20, markeredgecolor='black', 
                markeredgewidth=2, zorder=5)
        # 质心位置
        ax3.plot(centroid[0], centroid[1], 'g^', markersize=15, 
                markeredgecolor='black', markeredgewidth=2, zorder=5)
        # 连线
        ax3.plot([pos[0], centroid[0]], [pos[1], centroid[1]], 
                'k--', alpha=0.6, linewidth=2, zorder=3)
        # 箭头
        dx, dy = centroid[0] - pos[0], centroid[1] - pos[1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:  # 只在有显著位移时画箭头
            ax3.arrow(pos[0], pos[1], dx*0.8, dy*0.8,
                     head_width=0.06, head_length=0.06, 
                     fc='blue', ec='black', linewidth=1.5, 
                     alpha=0.7, zorder=4)
        # 标签
        ax3.text(pos[0] + 0.05, pos[1] + 0.08, f'A{i}', ha='left', 
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax3.text(centroid[0] + 0.05, centroid[1] - 0.08, f'C{i}', ha='left', 
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    
    # 4. 熵地图 + Voronoi边界 + 智能体和质心
    ax4 = axes[1, 1]
    im4 = ax4.imshow(entropy_map.T, 
                     origin='lower', cmap='RdYlGn_r', alpha=0.9,  # 使用相同色图
                     extent=[belief_map.world_min, belief_map.world_max,
                            belief_map.world_min, belief_map.world_max],
                     vmin=entropy_min, vmax=entropy_max)  # 使用实际范围
    ax4.set_title('综合视图: 熵分布 + Voronoi + 质心向量\n质心偏向高熵(高不确定性)区域', 
                  fontsize=11, fontweight='bold')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    
    # 添加Voronoi边界
    ax4.contour(voronoi_map.T, levels=np.unique(voronoi_map), 
                colors='cyan', linewidths=2.5, alpha=0.9,
                extent=[belief_map.world_min, belief_map.world_max,
                       belief_map.world_min, belief_map.world_max])
    
    # 标记智能体和质心
    for i, (pos, centroid) in enumerate(zip(agent_positions, centroids)):
        # 智能体
        ax4.plot(pos[0], pos[1], 'w*', markersize=20, 
                markeredgecolor='black', markeredgewidth=2, zorder=5)
        # 质心
        ax4.plot(centroid[0], centroid[1], marker='^', color='lime', 
                markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=5)
        # 箭头向量
        dx, dy = centroid[0] - pos[0], centroid[1] - pos[1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            ax4.arrow(pos[0], pos[1], dx*0.85, dy*0.85,
                     head_width=0.06, head_length=0.06, 
                     fc='yellow', ec='black', linewidth=2, 
                     alpha=0.8, zorder=4)
        # 标签
        ax4.text(pos[0], pos[1] + 0.09, f'A{i}', ha='center', 
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.colorbar(im4, ax=ax4, label='Shannon Entropy')
    ax4.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.savefig('global_belief_map_test.png', dpi=200, bbox_inches='tight')
    print("\n可视化图像已保存到: global_belief_map_test.png")
    plt.show()
    print()

def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("GlobalBeliefMap 功能测试")
    print("=" * 50 + "\n")
    
    test_basic_initialization()
    test_coordinate_conversion()
    test_bayesian_update()
    test_multiple_agents()
    test_voronoi_and_centroids()
    test_high_uncertainty_regions()
    visualize_belief_map()
    
    print("=" * 50)
    print("所有测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()