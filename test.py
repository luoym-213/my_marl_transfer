import torch

import torch

# 设置 m 和 n 的值
m = 2
n = 3
dist = 7

# 生成一个示例矩阵，内容从 0 到 17，共 6 行，每行 3 个元素
matrix = torch.arange(m * n * 2 * n).float().view(m * n, 2*n)
print("原始矩阵 (shape: {}):".format(matrix.shape))
print(matrix)

matrix = torch.norm(matrix.contiguous().view(m * n, n, 2), p=2, dim=2)
print("\n计算每两个数的欧几里得距离后的结果 (shape={}):\n{}".format(matrix.shape, matrix))

matrix = matrix > dist
print("matrix",matrix)

# 重新构造张量：

# 2. 交换前两个维度，得到 (m, n, n)
result = matrix.contiguous().view(n, m, n).permute(1, 0, 2)

print("\n重构后的矩阵 (shape: {}):".format(result.shape))
print(result)

mask = torch.full(size=(m,n,n),fill_value=0,dtype=torch.uint8)
mask.copy_(result)
print("mask",mask)
print("mask shape",mask.shape)