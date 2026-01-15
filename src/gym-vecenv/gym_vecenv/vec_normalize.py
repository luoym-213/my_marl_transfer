from .vec_env import VecEnvWrapper
from .running_mean_std import RunningMeanStd
import numpy as np

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

class MultiAgentVecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, 
                 ob=True, ret=True, clipob=10., cliprew=10., 
                 high_ret=False, cliphighrew=50.,   # high reward normalization not used in practice
                 gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.n = len(self.observation_space)
        self.ob_rms = [RunningMeanStd(shape=x.shape) for x in self.observation_space] if ob else None
        self.ret_rms = RunningMeanStd(shape=(self.n,)) if ret else None
        self.high_ret_rms = RunningMeanStd(shape=(self.n,)) if high_ret else None

        self.clipob = clipob
        self.cliprew = cliprew
        self.cliphighrew = cliphighrew

        self.ret = np.zeros((self.num_envs, self.n))
        self.highret = np.zeros((self.num_envs, self.n))

        self.gamma = gamma
        self.epsilon = epsilon
        

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a dict:
            news['agent'] is a boolean vector indicating whether each element is new.
        """
        obs, rews, high_rews, news, infos, sta = self.venv.step_wait()
        
        # 更新累计折扣回报
        self.ret = self.ret * self.gamma + rews
        if self.high_ret_rms:
            self.highret = self.highret * self.gamma + high_rews
        
        # 归一化观测
        obs = self._obfilt(obs)
        
        # 归一化奖励
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        if self.high_ret_rms:
            self.high_ret_rms.update(self.highret)
            high_rews = np.clip(high_rews / np.sqrt(self.high_ret_rms.var + self.epsilon), -self.cliphighrew, self.cliphighrew)
        
        # 🔥 重要修复：检测episode结束并重置累计回报
        # news['agent'] 是每个环境中每个智能体的done状态
        # shape: (num_envs, n_agents) 或 list of lists
        agent_dones = None
        if isinstance(news, dict) and 'agent' in news:
            agent_dones = np.array(news['agent'])
        elif isinstance(news, (list, np.ndarray)) and len(news) > 0 and isinstance(news[0], dict) and 'agent' in news[0]:
            agent_dones = np.array([d['agent'] for d in news])

        if agent_dones is not None:
            # 判断每个环境是否所有智能体都结束
            # all(axis=1) 表示对每个环境检查是否所有智能体都为True
            episode_dones = np.all(agent_dones, axis=1)  # shape: (num_envs,)
            
            # 对于完全结束的环境，重置累计回报
            self.ret[episode_dones] = 0.0
            self.highret[episode_dones] = 0.0
        
        return obs, rews, high_rews, news, infos, sta

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.reshape(obs,(self.num_envs,self.n,-1))
            for j in range(self.n):
                self.ob_rms[j].update(obs[:,j])
                t = np.clip((np.array(list(obs[:,j]),dtype=np.float) - self.ob_rms[j].mean) / np.sqrt(self.ob_rms[j].var + self.epsilon), -self.clipob, self.clipob)
                t = np.reshape(t,(self.num_envs,-1))
                for k in range(t.shape[0]):
                    obs[:,j][k] = t[k]
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs, sta, reset_info = self.venv.reset()
        return self._obfilt(obs), sta, reset_info