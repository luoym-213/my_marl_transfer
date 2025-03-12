import pandas as pd
import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm


parser = argparse.ArgumentParser(description='绘制两个CSV文件的Step vs Value曲线图。')
parser.add_argument('file_paths', nargs='+', help='一个或多个CSV文件路径 (例如：suc1.csv suc1.5.csv suc2.csv ...)')
parser.add_argument('--labels', nargs='+', help='一个或多个曲线名称')
parser.add_argument('--drawtype', default='success', help='which type of data you want to draw, default is success rate)')
parser.add_argument('--smooth', action='store_true', help='是否对曲线进行平滑处理')

args = parser.parse_args()
# 定义CSV文件路径
file_paths = args.file_paths
labels = args.labels

# 定义线条颜色
colors = cm.get_cmap('tab10').colors

# 绘制曲线图
plt.figure(figsize=(10, 6)) # 可选：设置图形大小

# 方法 1: 使用 matplotlib.rcParams 设置全局字体'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题 (中文)

for i, file_path in enumerate(file_paths):
    # ... (循环体) ...

    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
        df = df[df['Step'] <= 8000]
    except FileNotFoundError:
        print("错误：CSV文件未找到。",file_path)
        exit()

    # 检查CSV文件是否包含 'Step' 和 'Value' 列
    if 'Step' not in df.columns or 'Value' not in df.columns:
        print("错误：CSV文件缺少 'Step' 或 'Value' 列。")
        exit()

    color = colors[i % len(colors)]
    if args.smooth:
        df['Value'] = df['Value'].rolling(window=2).mean()
    plt.plot(df['Step'], df['Value'], color=color, label=labels[i])


# 添加图例
plt.legend()

if args.drawtype == 'success':
    # 添加标题和轴标签 (可选)
    plt.title('successful rate(num/50)')
    plt.xlabel('Step')
    plt.ylabel('successful rate')

if args.drawtype == 'length':
    # 添加标题和轴标签 (可选)
    plt.title('episode length')
    plt.xlabel('Step')
    plt.ylabel('Value')

if args.drawtype == 'reward':
    # 添加标题和轴标签 (可选)
    plt.title('training reward')
    plt.xlabel('Step')
    plt.ylabel('Value')

if args.drawtype == 'dist':
    # 添加标题和轴标签 (可选)
    plt.title('final min dist')
    plt.xlabel('Step')
    plt.ylabel('Value')

# 添加网格 (可选，使图形更易读)
plt.grid(True)

print("图形绘制完成。")
# 显示图形
plt.show()