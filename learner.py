import numpy as np
import torch
from rlcore.algo import JointPPO
from rlagent import Neo
from mpnn import MPNN
from utils import make_multiagent_env
from diaf import DIAF

def setup_master(args, env=None, return_env=False):
    if env is None:
        env = make_multiagent_env(args.env_name, num_agents=args.num_agents, dist_threshold=args.dist_threshold, 
                                  arena_size=args.arena_size, identity_size=args.identity_size)
    policy1 = None
    policy2 = None
    team1 = []
    team2 = []

    num_adversary = 0
    num_friendly = 0
    for i,agent in enumerate(env.world.policy_agents):
        if hasattr(agent, 'adversary') and agent.adversary:
            num_adversary += 1
        else:
            num_friendly += 1

    # share a common policy in a team
    action_space = env.action_space[i]
    entity_mp = args.entity_mp
    if args.env_name == 'simple_spread':
        num_entities = args.num_agents
    elif args.env_name == 'simple_formation':
        num_entities = 1
    elif args.env_name == 'simple_line':
        num_entities = 2
    else:
        raise NotImplementedError('Unknown environment, define entity_mp for this!')

    if entity_mp:
        pol_obs_dim = env.observation_space[i].shape[0] - 2*(2*num_entities-1)
    else:
        pol_obs_dim = env.observation_space[i].shape[0]

    # index at which agent's position is present in its observation
    pos_index = args.identity_size + 2
    for i, agent in enumerate(env.world.policy_agents):
        obs_dim = env.observation_space[i].shape[0]

        if hasattr(agent, 'adversary') and agent.adversary:
            if policy1 is None:
                policy1 = MPNN(input_size=pol_obs_dim,num_agents=num_adversary,num_entities=num_entities,action_space=action_space,
                               pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp).to(args.device)
            team1.append(Neo(args,policy1,(obs_dim,),action_space))
        else:
            if policy2 is None:
                policy2 = MPNN(input_size=pol_obs_dim,num_agents=num_friendly,num_entities=num_entities,action_space=action_space,
                               pos_index=pos_index, mask_dist=args.mask_dist,mask_obs_dist=args.mask_obs_dist,entity_mp=entity_mp).to(args.device)
            team2.append(Neo(args,policy2,(obs_dim,),action_space))

    diaf_method = DIAF(args, num_agents=args.num_agents, num_entities=num_entities, input_size=pol_obs_dim).to(args.device)
    
    master = Learner(args, [team1, team2], [policy1, policy2], env, diaf_method)
    
    if args.continue_training:
        print("Loading pretrained model")
        master.load_models(torch.load(args.load_dir, map_location=torch.device('cpu'))['models'])

    if return_env:
        return master, env
    return master


class Learner(object):
    # supports centralized training of agents in a team
    def __init__(self, args, teams_list, policies_list, env, dm):
        self.teams_list = [x for x in teams_list if len(x)!=0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]
        self.trainers_list = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        self.device = args.device
        self.args = args
        self.env = env
        self.dm = dm

    @property
    def all_policies(self):
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        return self.policies_list[0].attn_mat

    def initialize_obs(self, obs):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[:,i,:]).float().to(self.device))
            agent.rollouts.to(self.device)

    def act(self, step):
        actions_list = []
        dm = self.dm
        for team, policy in zip(self.teams_list, self.policies_list):
            # concatenate all inputs
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

            mask = dm.calculate_mask(all_obs) # mask: [96,5]，通信范围内为true，否则为false，与后面的mpnn的判断逻辑恰好相反
            
            # 第一步或每50步时进行初始化
            if  step % 50 == 0:
                # 需要对两个last以及卡尔曼参数进行初始化
                dm.initialize_last(step, team, mask)
                dm.initialize_kalman(step, team)

            else: # 其他步骤进行提取融合
                # new_value: [96,14]，new_mask: [96,5]
                new_value, new_mask = dm.infer_and_fuse(all_obs, mask, team, step)
                # 重新设置agent.rollouts.obs[step]
                for i,agent in enumerate(team):
                    agent.rollouts.obs[step].copy_(new_value[self.args.num_processes*i:self.args.num_processes*(i+1), :])
            
            new_value = torch.cat([agent.rollouts.obs[step] for agent in team])
            new_mask = torch.cat([agent.rollouts.last_mask[step] for agent in team])
            
            # indices = [0, 32, 64]
            # for idx in indices:
            #     if idx < all_obs.size(0):  # 检查索引是否在有效范围内
            #         print(f"all_obs[{idx}]:", all_obs[idx][:10])

            props = policy.act(new_value, all_hidden, all_masks, new_mask, deterministic=False) # a single forward pass 

            # split all outputs
            n = len(team)
            all_value, all_action, all_action_log_prob, all_states = [torch.chunk(x, n) for x in props]
            for i in range(n):
                # 这里规定了所有agent的self.states, self.action, self.action_log_prob, self.value
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]
                team[i].states = all_states[i]
                actions_list.append(all_action[i].cpu().numpy())

        return actions_list

    def update(self):
        return_vals = []
        # use joint ppo for training each team
        for i, trainer in enumerate(self.trainers_list):
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            vals = trainer.update(rollouts_list)
            return_vals.append([np.array(vals)]*len(rollouts_list))
        
        return np.stack([x for v in return_vals for x in v]).reshape(-1,3)

    def wrap_horizon(self):
        for team, policy in zip(self.teams_list,self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[-1] for agent in team])
            last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])
            obs_mask = torch.cat([agent.rollouts.last_mask[-1] for agent in team]) # 提取最后的一个观测遮罩
            
            with torch.no_grad():
                next_value = policy.get_value(last_obs, last_hidden, last_masks, obs_mask)

            all_value = torch.chunk(next_value,len(team))
            for i in range(len(team)):
                team[i].wrap_horizon(all_value[i])  # 计算每个agent的reward

    def after_update(self):
        for agent in self.all_agents:
            agent.after_update()

    def update_rollout(self, obs, reward, masks):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t[:, i, :]
            agent.update_rollout(agent_obs, reward[:,i].unsqueeze(1), masks[:,i].unsqueeze(1))

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, recurrent_hidden_states, mask, step):
        # used only while evaluating policies. Assuming that agents are in order of team!
        obs1 = []
        obs2 = []
        all_obs = []
        dm = self.dm
        for i in range(len(obs)):
            agent = self.env.world.policy_agents[i]
            if hasattr(agent, 'adversary') and agent.adversary:
                obs1.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
            else:
                obs2.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        if len(obs1)!=0:
            all_obs.append(obs1)
        if len(obs2)!=0:
            all_obs.append(obs2)
                
        ego_obs = torch.cat(all_obs[0])
        obs_mask = dm.calculate_mask(ego_obs)
        if step == 0:
            # 第一步初始化
            dm.initialize_eval(ego_obs, obs_mask)
            new_value = ego_obs
            new_mask = obs_mask
        
        else: #提取融合
            new_value, new_mask = dm.eval_infer_and_fuse(ego_obs, obs_mask)

        actions = []
        # 注意这里测试改动的只适用于simple——spread的时候了
        for team,policy,obs in zip(self.teams_list,self.policies_list,all_obs):
            if len(obs)!=0:
                _,action,_,_ = policy.act(new_value.to(self.device),None,None,new_mask,deterministic=True)
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions)

    def set_eval_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.train()
