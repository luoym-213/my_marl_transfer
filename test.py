import torch

def merge_tensor(pd_coords, inp_coords):
    """
    占位融合函数：默认返回 inp_coords。
    可替换为自定义融合策略（如加权平均）。
    输入维度：[N, 3, 2]
    """
    return inp_coords  # 默认直接使用观测值

def fuse_data_tensor(pd, mask1, inp, mask2):
    """
    使用张量操作融合 pd 和 inp。
    
    参数:
        pd: [N, 6] tensor -> reshape to [N, 3, 2]
        mask1: [N, 3] tensor
        inp: [N, 6] tensor -> reshape to [N, 3, 2]
        mask2: [N, 3] tensor

    返回:
        ninp: [N, 6] tensor
        mask3: [N, 3] tensor
    """ 
    N = pd.shape[0]
    pd = pd.view(N, 3, 2)
    print("pd:", pd[0:3])
    inp = inp.view(N, 3, 2)
    print("inp:", inp[0:3])
    print("mask1:", mask1[0:3])
    print("mask2:", mask2[0:3])

    # 初始化输出
    ninp = torch.zeros_like(pd)
    mask3 = torch.zeros_like(mask1)

    cond1 = (mask1 == 0) & (mask2 == 0)
    cond2 = (mask1 == 0) & (mask2 != 0)
    cond3 = (mask1 != 0) & (mask2 == 0)
    cond4 = (mask1 != 0) & (mask2 != 0)

    # 情况1：mask1=0, mask2=0 -> ninp=inp, mask3=mask1
    ninp[cond1] = inp[cond1]
    mask3[cond1] = mask1[cond1]


    # 情况2：mask1=0, mask2!=0 -> ninp=inp, mask3=1
    ninp[cond2] = inp[cond2]
    mask3[cond2] = 1

    # 情况3：mask1!=0, mask2==0
    # a. mask1 <= 5 -> ninp=pd, mask3=mask1+1
    cond3a = cond3 & (mask1 <= 5)
    ninp[cond3a] = pd[cond3a]
    mask3[cond3a] = mask1[cond3a] + 1

    # b. mask1 > 5 -> ninp=pd, mask3=0
    cond3b = cond3 & (mask1 > 5)
    ninp[cond3b] = pd[cond3b]
    mask3[cond3b] = 0

    # 情况4：mask1!=0, mask2!=0 -> ninp=merge(pd, inp), mask3=mask1
    merged = merge_tensor(pd[cond4], inp[cond4])
    print("pd[cond4]:", pd[cond4].shape)
    ninp[cond4] = merged
    mask3[cond4] = mask1[cond4]

    return ninp.view(N, 6), mask3

pd = torch.randn(96, 6)
inp = torch.randn(96, 6)
mask1 = torch.randint(0, 7, (96, 3))  # 包括可能的 >5 情况
mask2 = torch.randint(0, 2, (96, 3))  # 0或1

ninp, mask3 = fuse_data_tensor(pd, mask1, inp, mask2)
