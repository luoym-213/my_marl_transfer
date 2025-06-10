# 多智能体强化学习仿真渲染与GIF输出功能说明

## 目录
1. [功能概述](#功能概述)
2. [实现原理](#实现原理)
3. [代码修改说明](#代码修改说明)
4. [使用方法](#使用方法)
5. [参数配置](#参数配置)
6. [输出文件结构](#输出文件结构)
7. [注意事项](#注意事项)
8. [故障排除](#故障排除)

## 功能概述

本功能为多智能体强化学习系统添加了仿真渲染和GIF动画输出能力，主要特性包括：

- **自动GIF生成**：将仿真过程自动保存为GIF动画文件
- **文件夹组织**：为每次评估创建独立的文件夹
- **选择性渲染**：支持只保存GIF而不显示实时窗口
- **注意力可视化**：可选择性显示智能体间的通信连线
- **自动执行**：无需手动干预，程序自动完成渲染和保存

## 实现原理

### 核心机制

1. **渲染模式分离**
   - `mode='rgb_array'`：获取图像数据用于GIF保存
   - `mode='human'`：显示实时渲染窗口

2. **帧收集机制**
   - 在每个仿真步骤中收集渲染帧
   - 使用`imageio`库将帧序列保存为GIF

3. **注意力机制可视化**
   - 显示智能体间的通信连线
   - 基于距离掩码控制连线显示

### 技术架构

```
评估函数调用 → 创建评估文件夹 → 仿真执行 → 帧收集 → GIF保存
     ↓              ↓              ↓         ↓        ↓
  计数器递增    文件夹命名      渲染调用   帧存储   文件输出
```

## 代码修改说明

### 1. eval.py 主要修改

#### 全局计数器
```python
# Global counter to track evaluate function calls
_evaluate_call_count = 0
```

#### 渲染逻辑分离
```python
# 分离GIF保存和窗口显示
should_save_gif = args.record_video and t < 5
should_show_window = render and t < 5

# GIF保存渲染
if should_save_gif:
    render_result = env.render(mode='rgb_array', attn=attn)
    if render_result:
        frames.append(render_result[0])

# 窗口显示渲染
if should_show_window:
    env.render(mode='human', attn=attn)
```

#### 文件夹创建
```python
if args.record_video:
    eval_folder = os.path.join(args.gif_save_path, f"evaluation_{current_eval_call}")
    os.makedirs(eval_folder, exist_ok=True)
```

#### GIF保存
```python
if should_save_gif and frames:
    gif_filename = f"{t+1}.gif"
    gif_path = os.path.join(eval_folder, gif_filename)
    imageio.mimsave(gif_path, frames, duration=0.1)
```

### 2. rendering.py 修复

#### pyglet兼容性修复
```python
# 修复pyglet 1.3.2版本兼容性问题
arr = np.fromstring(image_data.get_data('RGBA', buffer.width * 4), dtype=np.uint8)
```

### 3. requirements.txt 更新

添加必要的依赖包：
```
imageio==2.31.1
```

## 使用方法

### 基本使用

#### 1. 只保存GIF（推荐）
```bash
python eval.py --record-video --gif-save-path ./gifs
```

#### 2. 只显示实时窗口
```bash
python eval.py --render
```

#### 3. 同时显示窗口和保存GIF
```bash
python eval.py --render --record-video --gif-save-path ./gifs
```

### 高级配置

#### 显示注意力连线
```bash
python eval.py --record-video --masking --gif-save-path ./gifs
```

#### 自定义通信距离
```bash
python eval.py --record-video --masking --mask-dist 2.0 --gif-save-path ./gifs
```

#### 调整评估回合数
```bash
python eval.py --record-video --num-eval-episodes 50 --gif-save-path ./gifs
```

## 参数配置

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record-video` | flag | False | 启用GIF保存功能 |
| `--gif-save-path` | str | 'gifs' | GIF文件保存路径 |
| `--render` | flag | False | 显示实时渲染窗口 |
| `--masking` | flag | False | 显示注意力连线 |
| `--mask-dist` | float | 1.5 | 通信距离阈值 |

### 评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-eval-episodes` | int | 30 | 评估回合总数 |
| `--env-name` | str | 'simple_spread' | 环境名称 |

### GIF参数

| 参数 | 说明 |
|------|------|
| 帧率 | 10 FPS (每帧0.1秒) |
| 保存范围 | 前5个回合 |
| 文件格式 | GIF |
| 命名规则 | `{回合号}.gif` |

## 输出文件结构

### 文件夹组织

```
gifs/
├── evaluation_1/          # 第1次评估调用
│   ├── 1.gif             # 第1回合
│   ├── 2.gif             # 第2回合
│   ├── 3.gif             # 第3回合
│   ├── 4.gif             # 第4回合
│   └── 5.gif             # 第5回合
├── evaluation_2/          # 第2次评估调用
│   ├── 1.gif
│   ├── 2.gif
│   ├── 3.gif
│   ├── 4.gif
│   └── 5.gif
└── evaluation_3/          # 第3次评估调用
    ├── 1.gif
    ├── 2.gif
    ├── 3.gif
    ├── 4.gif
    └── 5.gif
```

### 文件命名规则

- **文件夹命名**：`evaluation_{调用次数}`
- **GIF文件命名**：`{回合号}.gif`
- **调用次数**：全局计数器，每次调用`evaluate`函数递增

## 注意事项

### 1. 性能考虑

- **内存使用**：每个回合的帧数据会占用内存，建议控制评估回合数
- **存储空间**：GIF文件可能较大，注意磁盘空间
- **渲染速度**：`mode='rgb_array'`比`mode='human'`更快

### 2. 兼容性

- **pyglet版本**：已修复pyglet 1.3.2兼容性问题
- **操作系统**：支持Windows、Linux、macOS
- **Python版本**：建议Python 3.7+

### 3. 功能限制

- **保存范围**：只保存前5个回合的GIF
- **渲染模式**：GIF保存和窗口显示可以独立控制
- **注意力显示**：需要启用`--masking`参数

## 故障排除

### 常见问题

#### 1. pyglet渲染错误
```
AttributeError: 'ImageData' object has no attribute 'data'
```
**解决方案**：已修复，使用`get_data()`方法替代`data`属性

#### 2. GIF文件为空
**可能原因**：
- 渲染失败
- 帧收集失败
- 文件权限问题

**解决方案**：
- 检查控制台输出
- 确认`args.record_video`为True
- 检查保存路径权限

#### 3. 连线不显示
**可能原因**：
- 未启用`--masking`参数
- 智能体距离超过`mask_dist`
- 注意力权重为0

**解决方案**：
- 添加`--masking`参数
- 调整`--mask-dist`值
- 检查注意力机制

#### 4. 窗口不显示
**可能原因**：
- 未启用`--render`参数
- 只设置了`--record-video`

**解决方案**：
- 同时使用`--render`和`--record-video`
- 或只使用`--render`进行实时显示

### 调试建议

1. **查看控制台输出**：程序会显示保存进度和错误信息
2. **检查文件结构**：确认文件夹和文件是否正确创建
3. **验证参数**：确保命令行参数正确设置
4. **测试环境**：先用简单配置测试功能

## 总结

本功能为多智能体强化学习系统提供了完整的仿真可视化解决方案，支持：

- ✅ 自动GIF生成和保存
- ✅ 灵活的渲染控制
- ✅ 清晰的文件夹组织
- ✅ 注意力机制可视化
- ✅ 良好的错误处理
- ✅ 详细的进度反馈

通过合理配置参数，可以满足不同的可视化需求，为智能体行为分析和算法调试提供有力支持。 