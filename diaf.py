import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import numpy as np
from rlcore.algo import JointPPO
from rlagent import Neo
from mpnn import MPNN

class DIAF(nn.Module):
    def __init__(self, args, num_agents, num_entities, input_size):
        super().__init__()
        self.args = args
        self.num_agents = num_agents
        self.num_entities = num_entities
        self.input_size = input_size
        self.obs_shape = self.input_size+(self.num_entities+self.num_agents-1)*2 
        # 确保所有张量都在 GPU 上创建
        self.eval_last_value = torch.zeros(self.num_agents, self.obs_shape, device=args.device) #[3,14]
        self.eval_last_mask = torch.zeros(self.num_agents, self.num_entities+self.num_agents-1, device=args.device) #[3,5]
        self.eval_kalman_vel = torch.zeros(self.num_agents, self.obs_shape, device=args.device) #[3,14]
        self.eval_kalman_P = torch.zeros(self.num_agents, self.num_entities+self.num_agents-1, 4, 4, device=args.device) #[3,5,4,4]

    def initialize_eval(self, obs, mask):
        # 确保输入数据在 GPU 上
        obs = obs.to(self.args.device)
        mask = mask.to(self.args.device)
        
        self.eval_last_value.copy_(obs)
        ego = obs[:,2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
        self.eval_last_value[:,4:] = obs[:,4:] + ego

        self.eval_last_mask.copy_((mask>0).float())

        self.eval_kalman_vel.zero_()

        diag_vals = torch.tensor([1.0, 1.0, 10.0, 10.0], device=self.args.device)
        self.eval_kalman_P.copy_(torch.diag(diag_vals).unsqueeze(0).repeat(self.num_agents, self.num_entities+self.num_agents-1, 1, 1))

    def calculate_mask(self, obs):
        # obs:[96,14]
        # 仅提取entity的obs
        # 新增实体过滤逻辑
        num = self.num_agents + self.num_entities - 1 # 除自身外的所有可观测对象
        mask_obs_dist = self.args.mask_obs_dist
        inp = obs[:, self.input_size:] # [96,10]
        # 将inp重塑为[processes, num, 2]
        entities = inp.contiguous().view(-1, num, 2)  # [96,5,2]    
        # 计算相对距离（L2范数）
        distances = torch.norm(entities, p=2, dim=2)  # [processes, num]      
        # 生成观测掩码
        valid_mask = (distances <= mask_obs_dist)

        return valid_mask
    
    def initialize_last(self, team, mask):
        for i,agent in enumerate(team):
            position_abs = agent.rollouts.obs[0].clone().to(self.args.device)  # 创建原始数据的副本并移到 GPU
            ego = agent.rollouts.obs[0][:, 2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
            position_abs[:, 4:] = agent.rollouts.obs[0][:, 4:].to(self.args.device) + ego # 确保数据在 GPU 上
            agent.rollouts.last_value[0].copy_(position_abs)
            ini_mask = (mask.contiguous().view(self.num_agents, self.args.num_processes, self.num_agents * 2 - 1) > 0).float().permute(1, 0, 2)
            agent.rollouts.last_mask[0].copy_(ini_mask[:,i,:].to(self.args.device))
    
    def initialize_kalman(self, team):
        for agent in team:
            agent.rollouts.kalman_vel[0].zero_()
            diag_vals = torch.tensor([1.0, 1.0, 10.0, 10.0], device=self.args.device)
            agent.rollouts.kalman_P[0].copy_(torch.diag(diag_vals).unsqueeze(0).repeat(self.args.num_processes, self.num_agents*2-1, 1, 1))

    def _update_kalman_rollout(self, team, step, vel, P):
        for i,agent in enumerate(team):
            agent.rollouts.kalman_vel[step].copy_(vel[self.args.num_processes*i:self.args.num_processes*(i+1), :])
            agent.rollouts.kalman_P[step].copy_(P[self.args.num_processes*i:self.args.num_processes*(i+1),:,:,:])
    
    def _update_last_rollout(self, team, step, value, mask):
        for i,agent in enumerate(team):
            agent.rollouts.last_value[step].copy_(value[self.args.num_processes*i:self.args.num_processes*(i+1), :])
            agent.rollouts.last_mask[step].copy_(mask[self.args.num_processes*i:self.args.num_processes*(i+1), :])
    
    def _update_eval(self, kalman_P, kalman_vel, last_value, last_mask):
        self.eval_last_value.copy_(last_value)

        self.eval_last_mask.copy_(last_mask)

        self.eval_kalman_vel.copy_(kalman_vel)

        self.eval_kalman_P.copy_(kalman_P)


    def _infer(self, value, vel, P):
        # 计算value的推理值
        # value:[processes*num_entities,num_entities*2]
        # vel:[processes*num_entities,num_entities*2]
        # P:[processes*num_entities,3,4,4]
        # 假设状态向量为 [x, y, vx, vy]
        # 状态转移矩阵 F
        dt = 1.0  # 时间步长
        F = torch.tensor([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=value.device)
        
        # 过程噪声协方差矩阵 Q
        q = 0.1  # 过程噪声强度
        Q = torch.tensor([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ], device=value.device)
        
        batch_size = value.shape[0]
        num = value.shape[1] // 2
        
        # 重塑输入为 [batch_size, num_landmarks, 2]
        values = value.view(batch_size, num, 2)
        vels = vel.view(batch_size, num, 2)

        # 合并values和vels为[batch_size, num, 4]
        states = torch.cat([values, vels], dim=2)  # 在最后一个维度上拼接
        
        # 对每个landmark进行预测
        predicted_states = torch.matmul(F, states.unsqueeze(-1)).squeeze(-1) # [batch_size, num, 4]
        # 计算预测的协方差矩阵，# F @ P @ F.T -> [B, N, 4, 4]
        predicted_P = torch.einsum('ij,bnjk,kl->bnil', F, P, F.T)
        
        # 保留预测的位置坐标、速度
        predicted_positions = predicted_states[:, :, :2]
        predicted_vels = predicted_states[:, :, 2:]
        
        # 重塑回原始格式 [batch_size, num*2]
        predicted_values = predicted_positions.reshape(batch_size, -1)
        predicted_vels = predicted_vels.reshape(batch_size, -1)

        return predicted_values, predicted_P, predicted_vels
    
    def _merge_tensor(self, value, observed, P, vel):
        """
        使用卡尔曼滤波更新步骤融合状态
        参数:
            N: 所有满足cond4的数量
            value: [N, 2] 预测的位置
            observed: [N, 2] 实际观测值 [x, y]
            P: [N, 4, 4] 协方差矩阵
            vel: [N, 2] 速度
        返回:
            updated: [N, 4] 融合后的状态
        """
        # 将value和vel组合成状态张量
        state = torch.cat([value, vel], dim=1) # [N, 4]

        # 观测矩阵 H（只观测 x, y）
        H = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], device=value.device, dtype=torch.float32)  # [2, 4]

        # 观测的是位置，所以观测噪声为0.01
        R = torch.diag(torch.tensor([1e-5, 1e-5], device=value.device))  # [2, 2]

        # 创新（residual）
        z_pred = state @ H.T  # [N, 2]
        y = (observed - z_pred).unsqueeze(-1)  # [N, 2, 1]

        # 卡尔曼增益
        # 计算 H @ P -> [N, 2, 4]
        HP = torch.einsum('ij,njk->nik', H, P)
        # 计算 HP @ H.T -> [N, 2, 2]
        HPH = torch.einsum('nij,kj->nik', HP, H)
        # 计算 S = HP @ H.T + R -> [N, 2, 2]
        S = HPH + R
        # 计算 S的逆
        S_inv = torch.linalg.inv(S) # [N, 2, 2]
        # 计算 PHt = P @ H.T -> [N, 4, 2]
        PHt = torch.einsum('nij,kj->nik', P, H)
        # (P @ H.T) @ S^{-1} -> [N, 4, 2]
        K = torch.einsum('nij,njk->nik', PHt, S_inv)
        # 计算更新后的状态  [N, 4]+ [N, 4, 2] @ [N, 2, 1] -> [N, 4]
        updated = state + torch.einsum('nij,njk->nik', K, y).squeeze(-1)
        # 计算更新后的协方差矩阵
        P = torch.matmul(torch.eye(4, device=value.device) - torch.einsum('nij,jk->nik', K, H), P)
        
        return updated, P
    
    def _fuse(self, lv, lm, all_obs_abs, mask, lvel, lP):
        """
        使用张量操作融合 lv 和 all_obs_abs。
        
        参数:
            lv: [batchsize, num*2] tensor -> reshape to [batchsize, num, 2]
            lm: [batchsize, num] tensor
            all_obs_abs: [batchsize, num*2] tensor -> reshape to [batchsize, num, 2]
            mask: [batchsize, num] tensor
            lvel: [batchsize, num*2] tensor -> reshape to [batchsize, num, 2]
            lP: [batchsize, num, 4, 4] tensor

        返回:
            rv: [batchsize, num*2] tensor
            rm: [batchsize, num] tensor
            rvel: [batchsize, num*2] tensor
            rP: [batchsize, num, 4, 4] tensor

        """
        # 检查lv和all_obs_abs的维度是否一致
        if lv.shape[1] != all_obs_abs.shape[1]:
            raise ValueError("lv和all_obs_abs的维度不一致")
        # 检查lm和mask的维度是否一致
        if lm.shape[1] != mask.shape[1]:
            raise ValueError("lm和mask的维度不一致")
        
        N = lv.shape[0]
        num = lv.shape[1] // 2
        lv = lv.view(N, num, 2)
        all_obs_abs = all_obs_abs.view(N, num, 2)
        lvel = lvel.view(N, num, 2)
        # 初始化输出
        rv = torch.zeros_like(lv)
        rm = torch.zeros_like(lm)
        rvel = torch.zeros_like(lv)
        rP = torch.zeros_like(lP)   # [N,num,4,4]
        # 条件分布:[N,num]
        
        cond1 = (lm == 0) & (mask == 0)
        cond2 = (lm == 0) & (mask != 0)
        cond3 = (lm != 0) & (mask == 0)
        cond4 = (lm != 0) & (mask != 0)

        # 情况1：lm=0, mask=0 -> rv=all_obs_abs, rm=lm
        rv[cond1] = all_obs_abs[cond1]
        rm[cond1] = lm[cond1]
        rvel[cond1].zero_()
        # 初始化rP为对角矩阵
        diag_vals = torch.tensor([1.0, 1.0, 10.0, 10.0], device=rv.device)
        rP[cond1] = torch.diag(diag_vals).unsqueeze(0).expand(cond1.sum(), -1, -1)
        
        # 情况2：lm=0, mask!=0 -> rv=all_obs_abs, rm=1
        rv[cond2] = all_obs_abs[cond2]
        rm[cond2] = 1
        rvel[cond1].zero_()
        # 初始化rP为对角矩阵
        diag_vals = torch.tensor([1.0, 1.0, 10.0, 10.0], device=rv.device)
        rP[cond1] = torch.diag(diag_vals).unsqueeze(0).expand(cond1.sum(), -1, -1)

        # 情况3：lm!=0, mask==0
        # a. lm <= 5 -> rv=lv, rm=lm+1
        cond3a = cond3 & (lm <= 5)
        # print("rv[cond3a][0]:", rv[cond3a][0])
        # print("lv[cond3a][0]:", lv[cond3a][0])
        rv[cond3a] = lv[cond3a]
        rm[cond3a] = lm[cond3a] + 1
        rvel[cond3a] = lvel[cond3a]
        rP[cond3a] = lP[cond3a]

        # b. lm > 5 -> rv=lv, rm=0
        cond3b = cond3 & (lm > 5)
        rv[cond3b] = lv[cond3b]
        rm[cond3b] = 0
        rvel[cond1].zero_()
        # 初始化rP为对角矩阵
        diag_vals = torch.tensor([1.0, 1.0, 10.0, 10.0], device=rv.device)
        rP[cond1] = torch.diag(diag_vals).unsqueeze(0).expand(cond1.sum(), -1, -1)

        # 情况4：lm!=0, mask!=0 -> rv=merge(lv, all_obs_abs), rm=lm
        updated, P = self._merge_tensor(lv[cond4], all_obs_abs[cond4], lP[cond4], lvel[cond4])
        rv[cond4] = updated[:, :2]
        rm[cond4] = 1
        rvel[cond4] = updated[:, 2:]
        rP[cond4] = P

        # 恢复出返回的形状
        # rv:[N,num,2] -> [N,num*2]
        # rvel:[N,num,2] -> [N,num*2]
        rv = rv.view(N, -1)
        rvel = rvel.view(N, -1)
        
        return rv, rm, rvel, rP

    def infer_and_fuse(self, all_obs, mask, team, step):
        last_value = torch.cat([agent.rollouts.last_value[step-1] for agent in team]) # [96,14]
        last_mask = torch.cat([agent.rollouts.last_mask[step-1] for agent in team])
        kalman_vel = torch.cat([agent.rollouts.kalman_vel[step-1] for agent in team]) # [96,14]
        kalman_P = torch.cat([agent.rollouts.kalman_P[step-1] for agent in team]) # [96,5,4,4]
        # 筛选出对应agent以及entity的value和mask
        landmark_value = last_value[:,self.input_size:self.input_size+self.num_entities*2]
        lm = last_mask[:,:self.num_entities]
        agent_value = last_value[:,self.input_size+self.num_entities*2:]
        am = last_mask[:,self.num_entities:]
        # 计算绝对位置
        all_obs_abs = all_obs.clone()  # 创建原始数据的副本
        ego = all_obs[:, 2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
        all_obs_abs[:, 4:] = all_obs[:, 4:] + ego # [96,14]，这里的观测数据已经是绝对位置
        # 提取上一步的预测速度
        landmark_vel = kalman_vel[:,self.input_size:self.input_size+self.num_entities*2]
        agent_vel = kalman_vel[:,self.input_size+self.num_entities*2:]
        # 提取上一步的协方差矩阵
        landmark_P = kalman_P[:,:self.num_entities,:,:]
        # agent_P = kalman_P[:,self.num_entities:,:,:]
        # 计算两种对象的推理值
        lv, lP, lvel = self._infer(landmark_value, landmark_vel, landmark_P)
        # av, aP, avel = self._infer(agent_value, agent_vel, agent_P)
        # 计算融合值
        return_lv, return_lm, return_vel, return_P = self._fuse(lv, lm, all_obs_abs[:,self.input_size:self.input_size+self.num_entities*2], mask[:,:self.num_entities], lvel, lP)
        # return_av, return_am = self._fuse(av, am, all_obs_abs[:,self.input_size+self.num_entities*2:], mask[:,self.num_entities:])

        # 保存当步的预测速度和预测协方差
        kalman_vel[:, self.input_size:self.input_size+self.num_entities*2].copy_(return_vel)  # [96, 6]
        kalman_P[:, :self.num_entities, :, :].copy_(return_P)  # [96, 3, 4, 4]
        self._update_kalman_rollout(team, step, kalman_vel, kalman_P)

        # 保存当步的融合值和mask，这里的融合值是绝对位置
        last_value[:, :self.input_size].copy_(all_obs_abs[:, :self.input_size])  # [96, 4]
        last_value[:, self.input_size:self.input_size+self.num_entities*2].copy_(return_lv)  # [96, 6]
        last_value[:, self.input_size+self.num_entities*2:].copy_(all_obs_abs[:, self.input_size+self.num_entities*2:])  # [96, 6]
        last_mask[:, :self.num_entities].copy_(return_lm)  # [96, 3]
        self._update_last_rollout(team, step, last_value, last_mask)

        # 计算相对距离
        # 计算ego位置
        return_obs_value = all_obs.clone()  # 创建原始数据的副本
        ego_pos = all_obs_abs[:, 2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
        return_obs_value[:,4:].copy_(last_value[:, self.input_size:] - ego_pos)  # [96, 10]
        
        return return_obs_value, last_mask
    
    def eval_infer_and_fuse(self,all_obs,mask):
        """
        input:
            obs: [3,14]
            mask:[3,5]
        """
        # 提取上一步内容
        last_value = self.eval_last_value.clone()
        last_mask = self.eval_last_mask.clone()
        kalman_vel = self.eval_kalman_vel.clone()
        kalman_P = self.eval_kalman_P.clone()
        # 筛选出对应agent以及entity的value和mask
        landmark_value = last_value[:,self.input_size:self.input_size+self.num_entities*2]
        lm = last_mask[:,:self.num_entities]
        agent_value = last_value[:,self.input_size+self.num_entities*2:]
        am = last_mask[:,self.num_entities:]
        # 计算绝对位置
        all_obs_abs = all_obs.clone()  # 创建原始数据的副本
        ego = all_obs[:, 2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
        all_obs_abs[:, 4:] = all_obs[:, 4:] + ego # [3,14]，这里的观测数据已经是绝对位置
        # 提取上一步的预测速度
        landmark_vel = kalman_vel[:,self.input_size:self.input_size+self.num_entities*2]
        agent_vel = kalman_vel[:,self.input_size+self.num_entities*2:]
        # 提取上一步的协方差矩阵
        landmark_P = kalman_P[:,:self.num_entities,:,:]
        # 计算两种对象的推理值
        lv, lP, lvel = self._infer(landmark_value, landmark_vel, landmark_P)
        # 计算融合值
        return_lv, return_lm, return_vel, return_P = self._fuse(lv, lm, all_obs_abs[:,self.input_size:self.input_size+self.num_entities*2], mask[:,:self.num_entities], lvel, lP)
        
        # 保存当步的预测速度和预测协方差,以及两个last
        kalman_vel[:, self.input_size:self.input_size+self.num_entities*2].copy_(return_vel)
        kalman_P[:, :self.num_entities, :, :].copy_(return_P)
        last_value[:, :self.input_size].copy_(all_obs_abs[:, :self.input_size])  # [3, 4]
        last_value[:, self.input_size:self.input_size+self.num_entities*2].copy_(return_lv)  # [3, 6]
        last_value[:, self.input_size+self.num_entities*2:].copy_(all_obs_abs[:, self.input_size+self.num_entities*2:])  # [3 6]
        last_mask[:, :self.num_entities].copy_(return_lm)  # [3 3]
        self._update_eval(kalman_P, kalman_vel, last_value, last_mask)
        
        # 计算返回的new_value，相对值  
        return_obs_value = all_obs.clone()  # 创建原始数据的副本
        ego_pos = all_obs_abs[:, 2:4].unsqueeze(1).repeat(1, self.num_entities+self.num_agents-1, 1).reshape(-1, 2 * (self.num_entities+self.num_agents-1))
        return_obs_value[:,4:].copy_(last_value[:, self.input_size:] - ego_pos)  # [96, 10]

        return return_obs_value, last_mask