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
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., high_ret=True, cliphighrew=50., gamma=0.99, epsilon=1e-8):
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

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, high_rews, news, infos, sta = self.venv.step_wait()
        # 更新累计折扣回报
        self.ret = self.ret * self.gamma + rews
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

        # ⭐ 重置已完成环境的累计回报
        # 如果任一 agent done，重置该环境的所有 agent 的 ret
        if news.ndim == 2:
            # 如果是多智能体，检查每个环境的所有智能体
            done_envs = np.any(news, axis=1, keepdims=True)  # (num_envs, 1)
            done_mask = np.repeat(done_envs, self.n, axis=1)  # (num_envs, n)
        else:
            # 单智能体情况
            done_mask = news.reshape(-1, 1)

        # 清零已完成环境的累计回报
        self.ret = self.ret * (1 - done_mask)
        self.highret = self.highret * (1 - done_mask)
        
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