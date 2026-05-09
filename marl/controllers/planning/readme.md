# RRT_GNN 使用说明

基于熵值和Voronoi区域的RRT探索算法，用于多智能体环境中的高价值目标探索。

## 1. 导入方式

### 方式1：使用相对路径（推荐，便于服务器间迁移）

```python
import sys
import os

# 获取当前文件所在目录的父目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 如果当前文件在项目根目录下的子目录
# 或者
project_root = os.path.dirname(os.path.dirname(current_dir))  # 如果当前文件在项目根目录下的子子目录

# 添加planning目录到路径
sys.path.append(os.path.join(project_root, 'planning'))

from rrt_GNN import RRT_GNN, compute_voronoi_regions
import numpy as np
```

### 方式2：使用绝对路径（需要根据实际部署路径修改）

```python
import sys
sys.path.append('/path/to/your/marl_transfer/planning')
from rrt_GNN import RRT_GNN, compute_voronoi_regions
import numpy as np
```

### 示例：从demos目录调用

如果您的文件位于 `demos/` 目录下：

```python
import sys
import os

# 获取项目根目录（demos的上一级）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 添加planning目录到路径
sys.path.append(os.path.join(project_root, 'planning'))

from rrt_GNN import RRT_GNN, compute_voronoi_regions
```

## 2. 单智能体使用

### 初始化参数

```python
rrt = RRT_GNN(
    start=[x, y],              # 必需：智能体当前位置（栅格索引）
    voronoi_mask=mask,         # 必需：[100,100]布尔数组，True表示该栅格在Voronoi区域内
    entropy_map=entropy,       # 必需：[100,100]浮点数组，每个栅格的香农熵值
    entropy_threshold=0.8,     # 可选：熵值采样阈值，默认0.8
    radius=5,                  # 可选：计算节点价值的半径（栅格单位），默认5
    expand_dis=3,              # 可选：RRT扩展步长（栅格单位），默认3
    max_iterations=500,        # 可选：最大迭代次数，默认500
    top_k=10                   # 可选：返回前K个高价值节点，默认10
)
```

### 参数说明

- **start**: `List[int]` - 智能体起始位置 `[x, y]`，为栅格索引坐标
- **voronoi_mask**: `np.ndarray` - 形状为 `[100, 100]` 的布尔数组，`True` 表示该栅格属于智能体的Voronoi区域
- **entropy_map**: `np.ndarray` - 形状为 `[100, 100]` 的浮点数组，范围 `[0, 1]`，表示每个栅格的香农熵值
- **entropy_threshold**: `float` - 只在熵值大于此阈值的区域进行采样，建议 `0.6-0.9`
- **radius**: `int` - 计算节点周围熵值累积的半径，建议 `3-10`
- **expand_dis**: `int` - RRT每次扩展的步长，建议 `2-5`
- **max_iterations**: `int` - RRT迭代次数，越大探索越充分但耗时越长
- **top_k**: `int` - 返回价值最高的前K个节点

### 执行规划

```python
result = rrt.planning()
```

### 返回值格式

返回一个列表，每个元素是一个字典：

```python
[
    {
        'position': (x, y),    # 节点位置（栅格索引）
        'value': 125.8         # 节点累积价值
    },
    {
        'position': (x2, y2),
        'value': 118.3
    },
    ...
]
```

返回值按 `value` 从高到低排序。

## 3. 多智能体使用

### 计算Voronoi区域划分

```python
# 智能体位置列表
agent_positions = [(25, 30), (70, 50), (45, 75)]

# 计算Voronoi区域
voronoi_masks = compute_voronoi_regions(
    agent_positions=agent_positions,  # 智能体位置列表
    map_size=(100, 100)               # 地图大小
)
```

### 为每个智能体执行规划

```python
all_results = []
for i, (agent_pos, voronoi_mask) in enumerate(zip(agent_positions, voronoi_masks)):
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
    
    candidates = rrt.planning()
    all_results.append({
        'agent_id': i,
        'position': agent_pos,
        'candidates': candidates
    })
```

## 4. 完整使用示例

```python
import numpy as np
from rrt_GNN import RRT_GNN, compute_voronoi_regions

# 准备数据
agent_position = [50, 50]
voronoi_mask = np.ones((100, 100), dtype=bool)  # 整个区域都是自己的
entropy_map = np.random.rand(100, 100)  # 熵值地图

# 创建RRT_GNN实例
rrt = RRT_GNN(
    start=agent_position,
    voronoi_mask=voronoi_mask,
    entropy_map=entropy_map,
    entropy_threshold=0.7,
    radius=5,
    expand_dis=3,
    max_iterations=500,
    top_k=10
)

# 执行规划
candidates = rrt.planning()

# 使用结果
print(f"找到 {len(candidates)} 个候选点")
best_target = candidates[0]['position']  # 最优目标点
print(f"最优目标: {best_target}, 价值: {candidates[0]['value']:.2f}")

# 获取所有候选点位置
all_positions = [c['position'] for c in candidates]
all_values = [c['value'] for c in candidates]
```

## 5. 注意事项

1. **坐标系统**：所有坐标都是栅格索引 `(x, y)`，其中 `x` 对应行，`y` 对应列
2. **数组形状**：`voronoi_mask` 和 `entropy_map` 必须是相同形状
3. **空采样列表**：如果Voronoi区域内没有高熵值区域（大于阈值），会抛出 `ValueError`，建议降低 `entropy_threshold`
4. **计算成本**：`max_iterations` 越大、`radius` 越大，计算时间越长
5. **返回数量**：实际返回的候选点数量可能少于 `top_k`（如果迭代生成的节点总数较少）

## 6. 性能建议

- **快速探索**：`entropy_threshold=0.9`, `max_iterations=200`, `radius=3`
- **平衡模式**：`entropy_threshold=0.8`, `max_iterations=500`, `radius=5`（推荐）
- **充分探索**：`entropy_threshold=0.7`, `max_iterations=1000`, `radius=7`

## 7. 算法原理

### 核心思想

RRT_GNN结合了快速探索随机树(RRT)和基于价值的节点评估，在智能体的Voronoi区域内寻找高熵值的探索目标：

1. **采样区域生成**：从Voronoi区域中筛选出香农熵大于阈值的栅格作为采样空间
2. **RRT扩展**：从采样空间随机选择目标点，逐步构建探索树
3. **节点价值计算**：每个节点的价值 = 自身周围半径r内的熵值累积和 + 父节点价值
4. **Top-K选择**：返回价值最高的K个节点作为候选探索目标

### 优势

- 在有限区域内高效探索
- 考虑累积收益，有利于发现高价值路径
- 支持多智能体协同探索
- 可调节探索-开发平衡（通过entropy_threshold和radius）

## 8. 可视化示例

运行以下命令查看可视化示例：

```bash
# 运行RRT_GNN内置示例
python /Users/ym/Public/codes/marl_transfer/planning/rrt_GNN.py

# 或运行demos目录下的外部调用示例
python /Users/ym/Public/codes/marl_transfer/demos/demo_rrt_GNN.py
```

**内置示例** (`planning/rrt_GNN.py`)：展示了3个智能体在各自Voronoi区域内的探索过程

**外部调用示例** (`demos/demo_rrt_GNN.py`)：演示如何从外部文件调用RRT_GNN，包括详细的步骤说明和结果分析

可视化内容包括：
- 香农熵热力图（背景）
- Voronoi区域边界（彩色线条）
- 智能体起始位置（星形标记）
- RRT探索树结构（半透明线条）
- Top-K候选点标注（带序号和价值）
