import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from gym.utils import seeding
from multiagent.global_info_map import GlobalInfoMap


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, state_callback=None, discrete_action=False, shared_viewer=True,
                 cam_range=1, enable_exploration_reward=True, mask_obs_dist=None
                 ):

        self.world = world
        # Set observation range if provided
        if mask_obs_dist is not None:
            self.world.mask_obs_dist = mask_obs_dist
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.state_callback = state_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = discrete_action
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.seed()
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.cam_range = cam_range
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

        # 初始化全局信息地图
        self.enable_exploration_reward = enable_exploration_reward
        if self.enable_exploration_reward:
            self.world_size = 2
            self.cell_size = 0.05
            self.global_info_map = GlobalInfoMap(world_size=self.world_size, cell_size=self.cell_size)
        else:
            self.global_info_map = None



    @property
    def episode_limit(self):
        return self.world.max_steps_episode

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()

        # 计算探索奖励 (在位置更新后)，并更新全局信息图，放在0维
        exploration_rewards = self._compute_exploration_rewards()

        # 碰撞惩罚、边界惩罚
        common_penaltie = self._compute_penaltie()

        all_explor_rewards = exploration_rewards + common_penaltie

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        state = self._get_state(self.world)

        # # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        all_gather_reward =  np.array(reward_n) + common_penaltie

        final_reward = np.concatenate([all_explor_rewards, all_gather_reward], axis=0)

        return obs_n, final_reward, done_n, info_n, state

    def reset(self):
        # reset world
        self.reset_callback(self.world)

        # 重置global_info_map
        # 根据智能体初始位置，预先更新地图
        if self.enable_exploration_reward:
            self.global_info_map.reset()

            for agent in self.world.policy_agents:
                fov_mask = self.global_info_map.get_fov_mask(agent.state.p_pos, self.world.mask_obs_dist)
                self.global_info_map.update_explored_area(fov_mask)

        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        state = self._get_state(self.world)
        return obs_n, state

    def _compute_exploration_rewards(self):
        """
        计算每个智能体的探索奖励
        
        返回:
            numpy array of shape (num_agents,) 包含每个智能体的探索奖励
        """
        if not self.enable_exploration_reward:
            return np.zeros(len(self.agents))
        
        num_agents = len(self.agents)
        rewards_explore = np.zeros(num_agents)
        
        # 1. 获取当前地图状态的副本 (更新前)
        grid_before_update = self.global_info_map.grid.copy()
        
        # 2. 准备累积本步所有新探索的区域
        newly_explored_this_step_mask = np.zeros_like(
            grid_before_update, dtype=bool
        )
        
        # 3. 获取所有智能体的位置
        agent_positions = np.array([agent.state.p_pos for agent in self.agents])
        
        # 4. 遍历每个智能体，计算其个体贡献
        for i in range(num_agents):
            agent_pos = agent_positions[i]
            
            # a. 计算该智能体的观测区域 (Field of View)
            fov_mask = self.global_info_map.get_fov_mask(agent_pos, self.world.mask_obs_dist)
            
            # b. 计算边际信息增益
            # 边际贡献 = 智能体的FoV 与 (旧地图中的未知区域) 的交集
            unknown_mask_before = (grid_before_update == 0)
            marginal_contribution_mask = fov_mask & unknown_mask_before
            
            # c. 计算个体奖励 (新发现的栅格数量)
            rewards_explore[i] = np.sum(marginal_contribution_mask)
            
            # d. 累积本步所有新探索的区域
            newly_explored_this_step_mask |= marginal_contribution_mask
        
        # 5. 一次性更新全局地图
        self.global_info_map.update_explored_area(newly_explored_this_step_mask)
        
        return rewards_explore
    
    def _compute_penaltie(self):
        num_agents = len(self.agents)
        penalties = np.zeros(num_agents)
        
        # 超参数
        SAFE_DISTANCE = 0.15  # 安全距离
        COLLISION_COEF = -20.0  # 碰撞惩罚系数
        BOUNDARY_PENALTY = -2.0  # 边界惩罚
        
        # 获取所有智能体的位置
        agent_positions = np.array([agent.state.p_pos for agent in self.agents])
        
        # 1. 计算碰撞惩罚
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # 计算智能体i和j之间的距离
                dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                
                # 如果距离小于安全距离，施加碰撞惩罚
                if dist < SAFE_DISTANCE:
                    collision_penalty = COLLISION_COEF * ((1 - dist / SAFE_DISTANCE) ** 2)
                    penalties[i] += collision_penalty
                    penalties[j] += collision_penalty
        
        # 2. 计算边界惩罚
        # 边界范围是 [-world_size/2, world_size/2]
        boundary = self.world_size / 2.0  # 默认是 2.0/2.0 = 1.0
        
        for i in range(num_agents):
            x, y = agent_positions[i]
            
            # 检查是否触碰到边界
            if abs(x) >= boundary or abs(y) >= boundary:
                penalties[i] += BOUNDARY_PENALTY
        
        return penalties
    
    
    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)
    
    # get state for environment
    def _get_state(self, world):
        if self.state_callback is None:
            return np.zeros(0)
        return self.state_callback(world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', attn=None):
        # attn: matrix of size (num_agents, num_agents) 

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            
            for entity in self.world.entities:
                # Add observation range circle for agents
                if 'agent' in entity.name and hasattr(self.world, 'mask_obs_dist'):
                    obs_range = rendering.make_circle(self.world.mask_obs_dist, filled=True)
                    obs_range_xform = rendering.Transform()
                    obs_range.set_color(0.5, 1.0, 0.5, alpha=0.1)
                    obs_range.add_attr(obs_range_xform)
                    self.render_geoms.append(obs_range)
                    self.render_geoms_xform.append(obs_range_xform)
                
                # Add agent/landmark circle
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            self.render_count = len(self.render_geoms)                
            # render attn graph
            if attn is not None:
                # initialize render geoms for line
                for i in range(self.n):
                    for j in range(i+1, self.n):
                        geom = rendering.Line(start=self.world.agents[i].state.p_pos,
                                              end=self.world.agents[j].state.p_pos,
                                              linewidth=2)
                        color = (1.0, 0.0, 0.0)
                        alpha = 0
                        geom.set_color(*color, alpha)
                        xform = rendering.Transform()
                        self.render_geoms.append(geom)
                        self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
        
        if attn is not None:
            self._add_lines(attn)

        # update geometry positions
        geom_idx = 0
        for entity in self.world.entities:
            # Update observation range circle position for agents
            if 'agent' in entity.name and hasattr(self.world, 'mask_obs_dist'):
                self.render_geoms_xform[geom_idx].set_translation(*entity.state.p_pos)
                geom_idx += 1
            
            # Update agent/landmark position
            self.render_geoms_xform[geom_idx].set_translation(*entity.state.p_pos)
            geom_idx += 1

        results = []
        for i in range(len(self.viewers)):
            if self.shared_viewer:
                cam_range = self.cam_range
                if self.discrete_action_space:
                    cam_range = cam_range * 1.0
                pos = np.zeros(self.world.dim_p)
                self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            else:
                cam_range = self.cam_range
                if self.discrete_action_space:
                    cam_range = cam_range * 1.0
                pos = self.agents[i].state.p_pos
                self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    def _add_lines(self, attn):
        k = self.render_count
        for i in range(self.n):
            for j in range(i+1, self.n):
                val = attn[i][j] + attn[j][i]
                geom = self.render_geoms[k]
                color = (1.0, 0.0, 0.0)
                # alpha proportional to mean attention
                # alpha = .5*val
                # binary masking
                alpha = val>0
                geom.set_color(*color, alpha)
                k += 1

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    def get_env_info(self):
        env_info = {"state_shape": self.get_state().shape[0],
                    # "state_shape": self.observation_space[0].shape[0],
                    "obs_shape": self.observation_space[0].shape[0],
                    "n_actions": self.action_space[0].n,
                    "n_agents": self.n,
                    "episode_limit": 50}
        return env_info

    def get_state(self):
        return np.concatenate([self._get_obs(agent) for agent in self.agents])
        # entity_pos = [entity.state.p_pos for entity in self.world.landmarks]
        # agent_pos = [other.state.p_pos for other in self.world.agents]
        # agent_vel = [other.state.p_vel for other in self.world.agents]
        # return np.concatenate(agent_pos+agent_vel+entity_pos)

    def get_avail_actions(self):
        return np.ones((self.n,5))

    def get_obs(self):
        return [self._get_obs(agent) for agent in self.agents]

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
