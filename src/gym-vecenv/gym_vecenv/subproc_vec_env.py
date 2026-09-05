import os
import warnings

import numpy as np
from multiprocessing import Process, Pipe
from .vec_env import VecEnv, CloudpickleWrapper


def _get_cpu_affinity():
    """Return the CPUs available to the parent process, when supported."""
    if not hasattr(os, 'sched_getaffinity'):
        return None

    try:
        cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        return None

    return cpu_affinity or None


def _restore_cpu_affinity(cpu_affinity):
    """Restore the parent's CPU set after multiprocessing's fork hooks run."""
    if cpu_affinity is None or not hasattr(os, 'sched_setaffinity'):
        return False

    requested_affinity = set(cpu_affinity)
    try:
        os.sched_setaffinity(0, requested_affinity)
        actual_affinity = set(os.sched_getaffinity(0))
    except OSError as error:
        warnings.warn(
            'Unable to restore environment worker CPU affinity: {}'.format(error),
            RuntimeWarning,
        )
        return False

    if actual_affinity != requested_affinity:
        warnings.warn(
            'Environment worker CPU affinity was restricted from {} to {}'.format(
                sorted(requested_affinity), sorted(actual_affinity)
            ),
            RuntimeWarning,
        )
        return False

    return True


# 每个子环境的独立进程，负责接收指令，执行环境操作，并返回结果。
def worker(remote, parent_remote, env_fn_wrapper, cpu_affinity=None):
    # Some PyTorch/OpenMP combinations narrow forked children to CPUs 0-1.
    # Restore the CPU set captured from the parent before constructing the env.
    _restore_cpu_affinity(cpu_affinity)
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, high_reward, done, info, sta = env.step(data)
            if np.all(done['agent']):
                ob, sta, info = env.reset()
            remote.send((ob, reward, high_reward, done, info, sta))
        elif cmd == 'reset':
            ob, sta, reset_info = env.reset()
            remote.send((ob, sta, reset_info))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_num_agents':
            num_agents = env.num_agents if hasattr(env, 'num_agents') else None
            remote.send(num_agents)
        elif cmd == 'get_available_actions':
            avlact = env.get_avail_actions()
            remote.send(avlact)
        else:
            raise NotImplementedError


# 主进程端容器，用于一次性控制多个 worker，实现并行环境执行
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        cpu_affinity = _get_cpu_affinity()
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote,
                                               CloudpickleWrapper(env_fn),
                                               cpu_affinity))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self._num_agents = [None]*nenvs
    
    @property
    def num_agents(self):
        return self._num_agents
        
    def _update_num_agents(self):
        # will be called only after env is reset
        for i, remote in enumerate(self.remotes):
            remote.send(('get_num_agents', None))
            self._num_agents[i] = remote.recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, high_rews, dones, infos, sta = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(high_rews), np.stack(dones), infos, np.stack(sta)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, sta, reset_info = zip(*results)
        obs = np.stack(obs)
        sta = np.stack(sta)
        # ret = np.stack([remote.recv() for remote in self.remotes])
        self._update_num_agents()
        return obs, sta, reset_info

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_available_actions(self):
        for i, remote in enumerate(self.remotes):
            remote.send(('get_available_actions',None))
        return np.stack([remote.recv() for remote in self.remotes])