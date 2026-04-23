# 高层决策 MLP 消融改动说明

## 1. 改动目的

本次改动用于消融实验：将原高层异构图决策（Graph/Attention）替换为可切换的 MLP 前馈决策，并保持训练框架（PPO/SMDP-PPO）与高层 Critic 不变，确保对比公平。

核心目标：

- 保留原 `graph` 高层 actor 路径，作为默认实现。
- 新增 `mlp` 高层 actor 路径，通过命令行开关切换。
- 保证 `get_high_level_goal`（采样阶段）与 `evaluate_high_actions`（训练评估阶段）使用一致的候选构建与分布计算逻辑。

---

## 2. 设计要点（本次落地版本）

### 2.1 固定长度输入模板

高层候选统一为固定长度槽位：

- `K` 个 explore 候选槽位
- `N` 个 landmark 槽位（`N = max_landmarks`）

统一后总长度为 `K + N`。

### 2.2 排序与语义分区（MLP 分支）

- Explore 槽位：按 utility/value 降序。
- Landmark 槽位：按与 ego 距离升序（仅在有效 landmark 上排序）。
- 两类槽位分区组织，不混排。

### 2.3 槽位特征

统一 slot 特征为：

`[dx, dy, utility, flag, d, cos, sin, type, valid]`

其中：

- Explore: `flag=occupied`, `type=0`
- Landmark: `flag=is_targeted`, `type=1`
- `valid` 为有效性标记（padding/无效位为 0）

### 2.4 掩码与安全处理

- Landmark 有效性由 `landmark_mask & edge_mask & not_targeted` 决定。
- 保留原逻辑：当某样本所有 landmark 均有效（表示已全部发现）时，关闭 explore 槽位。
- 若某样本出现全无效槽位，启用安全回退：强制第 1 个槽位有效，避免 Categorical 分布出现全 `-inf`。

### 2.5 训练评估一致性

`evaluate_high_actions` 中回映射 `goal -> selected_idx` 时，按 `task` 限制搜索域：

- `task=explore` 只在 explore 槽位中匹配
- `task=landmark` 只在 landmark 槽位中匹配

若对应域无有效槽位，则回退到全域有效槽位匹配。

---

## 3. 代码改动清单

### 3.1 参数与入口

1. `arguments.py`

- 新增参数：
  - `--high-actor-backbone {graph,mlp}`（默认 `graph`）
  - `--high-mlp-hidden`
  - `--high-mlp-layers`
  - `--high-mlp-dropout`

2. `learner.py`

- 在构造 `MPNN` 时传入上述新参数（adversary/friendly 两个 policy 路径均已透传）。

### 3.2 模型实现

3. `mpnn.py`

- `MPNN.__init__` 新增高层 actor 配置参数与合法性校验。
- 新增通用 `_build_mlp` 构建函数。
- 在 high-level 模块中新增 MLP 分支组件：
  - `mlp_context_encoder`
  - `mlp_slot_encoder`
  - `mlp_selection_head`
- 新增高层统一 helper：
  - `_sort_high_level_nodes_for_mlp`
  - `_build_unified_slot_tensors`
  - `_compute_graph_selection_logits`
  - `_compute_mlp_selection_logits`
  - `_build_high_selection_distribution`
  - `_map_goals_to_selected_indices`
- 重构 `get_high_level_goal`：改为调用统一分布构建逻辑。
- 重构 `evaluate_high_actions`：与上面复用完全一致的分布与回映射逻辑。

---

## 4. 使用方法

### 4.1 保持原图网络（默认）

```bash
python3 main.py --high-actor-backbone graph
```

### 4.2 启用 MLP 消融

```bash
python3 main.py \
  --high-actor-backbone mlp \
  --high-mlp-hidden 256 \
  --high-mlp-layers 3 \
  --high-mlp-dropout 0.0
```

---

## 5. 与已有 checkpoint 的兼容性说明

- 低层与高层 Critic 结构未改，兼容原流程。
- 高层 Actor 新增了 MLP 模块参数。
- 若加载旧 high-level actor checkpoint 到新模型，通常会出现 MLP 相关 key 缺失（`strict=False` 下可正常运行），用于 `graph` 路径通常不受影响。
- 进行 MLP 消融时，建议从头训练高层 actor（或单独准备 MLP actor 的预训练权重）。

---

## 6. 本地校验情况

已完成：

- 语法级校验：`python3 -m py_compile arguments.py learner.py mpnn.py`

未完成：

- 运行级前向 smoke test（当前环境缺少 `torch`，无法在本机执行 runtime 张量测试）。

---

## 7. 影响范围

本次仅修改以下文件：

- `arguments.py`
- `learner.py`
- `mpnn.py`

未修改 PPO 主训练逻辑、Rollout 存储结构与低层策略接口。
