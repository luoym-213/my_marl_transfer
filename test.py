import torch

def remove_diagonal(A):
    """
    移除邻接矩阵 A 的对角线元素。
    
    Args:
        A (torch.Tensor): 输入的邻接矩阵，形状为 [batchsize, n, n]
        
    Returns:
        torch.Tensor: 移除对角线后的矩阵，形状为 [batchsize, n, n-1]
    """
    batchsize, n, _ = A.shape
    mask = ~torch.eye(n, n, device=A.device, dtype=torch.bool)
    return A.masked_select(mask).view(batchsize, n, n-1)

def main():
    # 定义批量大小和矩阵大小
    batchsize = 2
    n = 3
    
    # 创建一个示例邻接矩阵 A，形状为 [batchsize, n, n]
    A = torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
    ], dtype=torch.float)
    
    print("原始邻接矩阵 A:")
    print(A)
    print("形状:", A.shape)  # [2, 3, 3]
    
    # 移除对角线元素
    A_new = remove_diagonal(A)
    
    print("\n移除对角线后的矩阵 A_new:")
    print(A_new)
    print("形状:", A_new.shape)  # [2, 3, 2]
    
    # 进一步验证结果
    # 对第一个批次
    print("\n第一个批次，移除对角线前:")
    print(A[0])
    print("第一个批次，移除对角线后:")
    print(A_new[0])
    
    # 对第二个批次
    print("\n第二个批次，移除对角线前:")
    print(A[1])
    print("第二个批次，移除对角线后:")
    print(A_new[1])

if __name__ == "__main__":
    main()