import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from gym.utils import seeding
from multiagent.global_info_map import GlobalInfoMap
from multiagent.global_belief_map import GlobalBeliefMap



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

        # landmarks 位置
        self.landmark_positions = [landmark.state.p_pos for landmark in self.world.landmarks]

        # 初始化已访问目标集合
        self.visited_landmarks = set()
        
        # 初始化智能体退役状态列表
        self.agents_done = [False] * self.n

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

        # 初始化全局信念地图
        self.enable_exploration_reward = enable_exploration_reward
        if self.enable_exploration_reward:
            self.world_size = 2
            self.cell_size = 0.02
            self.global_belief_map = GlobalBeliefMap(world_size=self.world_size, cell_size=self.cell_size, landmark_positions=self.landmark_positions, landmark_radius=0.05)
        else:
            self.global_belief_map = None



    @property
    def episode_limit(self):
        return self.world.max_steps_episode

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, data, goal_n=None):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': [], 'map': [], 'world_steps': self.world.steps}
        done = {}
        self.agents = self.world.policy_agents
        last_reward_n = []
        current_reward_n = []
        last_goal_n = []
        action_n = data['agents_actions']
        goal_n = data['agents_goals']
        task_n = data['agents_tasks']

        # 获取step前全局状态，智能体速度、位置，landmark位置
        state = self._get_state(self.world)

        # set action and goal for each agent
        for i, agent in enumerate(self.agents):
            # 如果智能体已经退役，设置其速度为0，不执行新动作
            if self.agents_done[i]:
                agent.state.p_vel = np.zeros(self.world.dim_p)
                agent.action.u = np.zeros(self.world.dim_p)
            else:
                ## 设置agent.action,留给world.step()使用
                self._set_action(action_n[i], agent, self.action_space[i])
                ## 设置agent.state.g_pos, 供reward计算使用
                self._set_goal(goal_n[i], agent)
        
        # 计算step前的距离奖励，即上一步智能体距离目标点的距离，以便后续计算差分奖励
        for agent in self.agents:
            last_reward_n.append(self._get_goal_reward(agent))

        # advance world state
        self.world.step()

        # 收集当前智能体位置
        agents_pos = np.array([a.state.p_pos for a in self.agents])

        # 必须在更新全图信息图前获取高层奖励，因为高层奖励依赖于step前的全局信息图
        agents_explore_rewards = self.global_belief_map.get_agent_step_explore_entropy(agents_pos, self.world.mask_obs_dist)
        agents_discover_target_rewards = self.global_belief_map.get_agent_discover_target_reward(agents_pos, self.world.mask_obs_dist)
        # 到达目标点奖励，需要满足当前当前task = 1，即collect模式，且距离目标点小于阈值
        agents_reach_target_rewards = self.get_target_reward(agents_pos, task_n, self.landmark_positions)
        total_high_rewards = np.array(agents_explore_rewards) + np.array(agents_discover_target_rewards) + np.array(agents_reach_target_rewards)
        # print(total_high_rewards.shape)

        # 根据获取的全局状态更新全局信息图
        if self.enable_exploration_reward:
            self.global_belief_map.update_beliefs(agents_pos, self.world.mask_obs_dist)

        # 获取step更新后的voronoi加权质心以及目标位置
        centroids = self.global_belief_map.compute_entropy_weighted_centroids(agents_pos) if self.enable_exploration_reward else None
        target_positions = self.global_belief_map.get_target_positions() if self.enable_exploration_reward else None

        # 添加world_steps到info_n中
        info_n['world_steps'] = self.world.steps

        # 将centroids和target_positions添加到info_n中
        info_n['map'].append(centroids)
        info_n['map'].append(target_positions)
        # 将高层策略需要的通道图、是否达到目标点分别加入
        info_n['belief_map'] = self.global_belief_map.belief_grid
        info_n['entropy_map'] = self.global_belief_map.compute_shannon_entropy()
        info_n['voronoi_masks'] = self.global_belief_map.get_voronoi_region_masks(agents_pos)
        info_n['distance_fields'] = self.global_belief_map.get_distance_fields(agents_pos, normalize=True)
        info_n['goal_done'] = self._get_goal_dones(self.agents)
        info_n['heatmap'] = self.global_belief_map.get_agents_heatmap(agents_pos,0.05)
        info_n['landmark_heatmap'] = self.global_belief_map.landmark_heatmap

        # 碰撞惩罚、边界惩罚
        common_penaltie = self._compute_penaltie()

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_goal_reward(agent))
            done_n.append(self._get_done(agent)) # 相当于整个episode是否结束
            info_n['n'].append(self._get_info(agent))

        # 获取全局状态，智能体速度、位置，landmark位置
        state = self._get_state(self.world)

        # # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        # 差分奖励
        reward_n = np.array(reward_n) - np.array(last_reward_n)

        # 势能奖励+碰撞惩罚
        all_reward =  reward_n + common_penaltie
        
        # 对已退役的智能体，将其奖励设置为0
        for i in range(len(all_reward)):
            if self.agents_done[i]:
                all_reward[i] = 0.0
                total_high_rewards[i] = 0.0

        done['all'] = done_n
        done['agent'] = self.agents_done

        return obs_n, all_reward, total_high_rewards, done, info_n, state

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.landmark_positions = [landmark.state.p_pos for landmark in self.world.landmarks]

        # 重置已访问目标集合
        self.visited_landmarks = set()
        
        # 重置智能体退役状态
        self.agents_done = [False] * self.n

        # 重置global_belief_map
        # 根据智能体初始位置，预先更新地图
        if self.enable_exploration_reward:
            self.global_belief_map.reset(self.landmark_positions)
            self.global_belief_map.update_beliefs(np.array([a.state.p_pos for a in self.world.policy_agents]), self.world.mask_obs_dist)

        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        reset_info = {'map': [], 'world_steps': self.world.steps} # 用于存储reset时的信息
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        state = self._get_state(self.world)

        # 获取当前智能体的位置，[num_agents, 2]
        agent_positions = np.array([agent.state.p_pos for agent in self.agents])
        # 获取reset后的voronoi加权质心以及目标位置
        centroids = self.global_belief_map.compute_entropy_weighted_centroids(agent_positions) if self.enable_exploration_reward else None
        target_positions = self.global_belief_map.get_target_positions() if self.enable_exploration_reward else None
        reset_info['map'].append(centroids)
        reset_info['map'].append(target_positions)

        # 将高层策略需要的通道图、是否达到目标点分别加入
        
        reset_info['belief_map'] = self.global_belief_map.belief_grid
        reset_info['entropy_map'] = self.global_belief_map.compute_shannon_entropy()
        reset_info['voronoi_masks'] = self.global_belief_map.get_voronoi_region_masks(agent_positions)
        reset_info['distance_fields'] = self.global_belief_map.get_distance_fields(agent_positions, normalize=True)
        reset_info['heatmap'] = self.global_belief_map.get_agents_heatmap(agent_positions, 0.05)
        reset_info['landmark_heatmap'] = self.global_belief_map.landmark_heatmap
        # reset下，goal_done全部为True
        reset_info['goal_done'] = [True] * len(self.agents)

        return obs_n, state, reset_info
    
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
    
    def _get_goal_reward(self, agent):
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
    
    # set env goal for a particular agent
    def _set_goal(self, goal, agent):
        agent.state.g_pos = goal

    def _get_goal_dones(self, agents):
        # check if high-level goal is achieved
        # agent pos: agent.state.p_pos
        # goal pos: agent.state.g_pos
        # threshold: self.world.dist_thres
        goal_dones = [np.linalg.norm(agent.state.p_pos - agent.state.g_pos) < self.world.dist_thres for agent in agents] 
        # 如果agent已经退役，则goal_done也设为False
        for i in range(len(goal_dones)):
            if self.agents_done[i]:
                goal_dones[i] = False
        return goal_dones

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', attn=None, goals=None, show_voronoi=False, info=None, show_uncertainty=True):
        # attn: matrix of size (num_agents, num_agents)
        # goals: array of shape (num_agents, 2) - goal positions for each agent
        # show_voronoi: bool - whether to show voronoi centroids
        # show_uncertainty: bool - whether to show uncertainty heatmap
        # info: dict - contains voronoi centroids information

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
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
        
        # 🔧 在每次渲染前，移除viewer中所有临时几何体
        temp_geoms_start_idx = len(self.render_geoms)
        
        # 从viewer中移除旧的临时几何体
        for viewer in self.viewers:
            viewer.geoms = viewer.geoms[:self.render_count]
        
        # 🎨 绘制不确定性热力图（作为最底层背景）
        if show_uncertainty and self.enable_exploration_reward and self.global_belief_map is not None:
            from multiagent import rendering
            
            # 可调参数：不确定性热力图样式
            UNCERTAINTY_ALPHA = 0.3       # 👈 修改这里：热力图透明度
            CELL_SIZE_SCALE = 1.0         # 👈 修改这里：栅格显示比例（相对于实际cell_size）
            
            # 获取香农熵地图
            entropy_map = self.global_belief_map.compute_shannon_entropy()
            
            # 归一化熵值到[0, 1]范围（最大熵为1.0）
            max_entropy = 1.0
            normalized_entropy = entropy_map / max_entropy
            
            # 获取地图参数
            world_min = self.global_belief_map.world_min
            cell_size = self.global_belief_map.cell_size
            map_dim = self.global_belief_map.map_dim
            
            # 绘制每个栅格的不确定性
            for i in range(map_dim):
                for j in range(map_dim):
                    uncertainty = normalized_entropy[i, j]
                    
                    # 跳过低不确定性的栅格（优化性能）
                    if uncertainty < 0.05:
                        continue
                    
                    # 计算栅格中心的世界坐标
                    x = world_min + (i + 0.5) * cell_size
                    y = world_min + (j + 0.5) * cell_size
                    
                    # 使用颜色映射：不确定性越高，颜色越深
                    # 这里使用红色到黄色的渐变
                    # 高不确定性（接近1）-> 深红色 (1, 0, 0)
                    # 低不确定性（接近0）-> 浅黄色 (1, 1, 0)
                    r = 1.0
                    g = 1.0 - uncertainty  # 不确定性越高，绿色分量越少
                    b = 0.0
                    
                    # 创建矩形表示栅格
                    rect_size = cell_size * CELL_SIZE_SCALE
                    rect = rendering.make_polygon([
                        (x - rect_size/2, y - rect_size/2),
                        (x + rect_size/2, y - rect_size/2),
                        (x + rect_size/2, y + rect_size/2),
                        (x - rect_size/2, y + rect_size/2)
                    ])
                    rect.set_color(r, g, b, alpha=UNCERTAINTY_ALPHA * uncertainty)
                    self.render_geoms.append(rect)
        
        # 绘制Voronoi图边界线
        if show_voronoi and info is not None:
            from multiagent import rendering
            
            # 获取所有智能体的位置
            agent_positions = np.array([agent.state.p_pos for agent in self.world.agents])
            
            if len(agent_positions) > 1:
                # 可调参数：Voronoi边界线样式
                VORONOI_LINE_WIDTH = 5
                VORONOI_LINE_COLOR = (0.7, 0.7, 0.7)
                VORONOI_LINE_ALPHA = 1
                
                # 使用GlobalBeliefMap的方法获取Voronoi边界
                if self.enable_exploration_reward and self.global_belief_map is not None:
                    voronoi_edges = self.global_belief_map.get_voronoi_edges(agent_positions)
                    
                    # 绘制Voronoi边界线
                    for edge in voronoi_edges:
                        start, end = edge
                        voronoi_line = rendering.Line(
                            start=(start[0], start[1]),
                            end=(end[0], end[1]),
                            linewidth=VORONOI_LINE_WIDTH
                        )
                        voronoi_line.set_color(*VORONOI_LINE_COLOR, alpha=VORONOI_LINE_ALPHA)
                        self.render_geoms.append(voronoi_line)
        
        # 绘制目标点 (goals)
        if goals is not None:
            from multiagent import rendering
            
            # 可调参数：目标点大小和连线宽度
            GOAL_MARKER_SIZE = 0.01
            GOAL_LINE_WIDTH = 1
            
            for i, goal in enumerate(goals):
                if goal is not None and len(goal) == 2:
                    # 绘制目标点（小圆圈）
                    goal_marker = rendering.make_circle(GOAL_MARKER_SIZE, filled=True)
                    goal_marker.set_color(1.0, 0.0, 1.0, alpha=0.8)
                    xform = rendering.Transform()
                    xform.set_translation(goal[0], goal[1])
                    goal_marker.add_attr(xform)
                    self.render_geoms.append(goal_marker)
                    
                    # 绘制智能体到目标点的连线
                    agent_pos = self.world.agents[i].state.p_pos
                    goal_line = rendering.Line(
                        start=(agent_pos[0], agent_pos[1]),
                        end=(goal[0], goal[1]),
                        linewidth=GOAL_LINE_WIDTH
                    )
                    goal_line.set_color(1.0, 0.0, 1.0, alpha=0.3)
                    self.render_geoms.append(goal_line)
        
        # 绘制voronoi加权质心
        if show_voronoi and info is not None and 'map' in info and len(info['map']) > 0:
            from multiagent import rendering
            
            # 可调参数：质心大小和连线宽度
            CENTROID_MARKER_SIZE = 0.025
            CENTROID_LINE_WIDTH = 1
            
            centroids = info['map'][0]
            if centroids is not None:
                for i, centroid in enumerate(centroids):
                    if centroid is not None and len(centroid) == 2:
                        # 绘制质心（圆形标记）
                        centroid_marker = rendering.make_circle(CENTROID_MARKER_SIZE, filled=True)
                        centroid_marker.set_color(0.0, 1.0, 1.0, alpha=0.8)
                        xform = rendering.Transform()
                        xform.set_translation(centroid[0], centroid[1])
                        centroid_marker.add_attr(xform)
                        self.render_geoms.append(centroid_marker)
                        
                        # 绘制智能体到质心的连线
                        agent_pos = self.world.agents[i].state.p_pos
                        centroid_line = rendering.Line(
                            start=(agent_pos[0], agent_pos[1]),
                            end=(centroid[0], centroid[1]),
                            linewidth=CENTROID_LINE_WIDTH
                        )
                        centroid_line.set_color(0.0, 1.0, 1.0, alpha=0.3)
                        self.render_geoms.append(centroid_line)
        
        # 🔧 将新的临时几何体添加到viewer
        for viewer in self.viewers:
            for geom in self.render_geoms[temp_geoms_start_idx:]:
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
        
        # 🔧 清理临时几何体
        if temp_geoms_start_idx < len(self.render_geoms):
            self.render_geoms = self.render_geoms[:temp_geoms_start_idx]

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
    
    def get_target_reward(self, agents_pos, agents_task, landmarks_pos):
        """
        首先检查agents_task，只有在collect模式（1）下才计算目标奖励。
        如果智能体到达目标点，如果这个目标点未访问过，返回一个奖励值，否则返回0。
        判断准则，距离差小于world.dist_thres
        当智能体到达目标后，会被标记为退役状态，后续将停止运动并持续返回0奖励。
        
        参数:
            agents_pos: 智能体位置数组，shape (n_agents, 2)
            agents_task: 智能体任务列表，1表示collect模式，0表示explore模式
            landmarks_pos: 目标点位置列表，每个元素是 (x, y)
        
        返回:
            rewards: 列表，每个智能体的目标到达奖励
        """
        # 初始化已访问目标集合（需要在 __init__ 中定义）
        if not hasattr(self, 'visited_landmarks'):
            self.visited_landmarks = set()
        
        rewards = []
        TARGET_REWARD = 10.0  # 到达新目标的奖励值
        
        for agent_idx, agent_pos in enumerate(agents_pos):
            agent_reward = 0.0
            
            # 如果智能体已经退役，直接返回0奖励
            if self.agents_done[agent_idx]:
                rewards.append(agent_reward)
                continue
            
            # 首先检查该智能体是否处于collect模式（task=1）
            agent_task = agents_task[agent_idx]
            if agent_task[0] != 1:  # 只有在collect模式下才计算目标奖励
                rewards.append(agent_reward)
                continue
            
            # 检查该智能体是否到达任何目标点
            for landmark_idx, landmark_pos in enumerate(landmarks_pos):
                # 计算到目标点的距离
                dist = np.linalg.norm(agent_pos - landmark_pos)
                
                # 如果距离小于阈值，认为到达目标
                if dist < self.world.dist_thres:
                    # 检查该目标是否已被访问过
                    if landmark_idx not in self.visited_landmarks:
                        # 首次访问该目标，给予奖励并标记为已访问
                        agent_reward = TARGET_REWARD
                        self.visited_landmarks.add(landmark_idx)
                        # 标记该智能体为退役状态
                        self.agents_done[agent_idx] = True
                        break  # 一个智能体在一个step只能获得一次目标奖励
            
            rewards.append(agent_reward)
        
        return rewards

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
