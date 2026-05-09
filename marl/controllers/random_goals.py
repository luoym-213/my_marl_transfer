import numpy as np


def sample_random_goals(env_state, num_processes, num_agents, arena_size, rng=None,
                        illegal_regions=None, max_attempts=128):
    """Sample legal random goals with shape [num_processes, num_agents, 2].

    The current MAPE environments do not define obstacle objects. The legality
    checks therefore enforce map bounds and optionally reject caller-provided
    circular illegal regions shaped as [(x, y, radius), ...].
    """
    del env_state
    rng = rng if rng is not None else np.random
    illegal_regions = illegal_regions or []
    goals = np.zeros((num_processes, num_agents, 2), dtype=np.float32)

    for proc_idx in range(num_processes):
        for agent_idx in range(num_agents):
            goals[proc_idx, agent_idx] = _sample_one_goal(
                rng,
                arena_size,
                illegal_regions,
                max_attempts,
            )
    return goals


def _sample_one_goal(rng, arena_size, illegal_regions, max_attempts):
    for _ in range(max_attempts):
        goal = rng.uniform(-arena_size, arena_size, size=2).astype(np.float32)
        if _is_legal_goal(goal, arena_size, illegal_regions):
            return goal

    return np.clip(goal, -arena_size, arena_size).astype(np.float32)


def _is_legal_goal(goal, arena_size, illegal_regions):
    if np.any(goal < -arena_size) or np.any(goal > arena_size):
        return False
    for x, y, radius in illegal_regions:
        if np.linalg.norm(goal - np.array([x, y], dtype=np.float32)) <= radius:
            return False
    return True


class RandomGoalManager:
    """Maintains low-level-only random goals and goal lifecycle statistics."""

    def __init__(self, num_processes, num_agents, arena_size, rng=None):
        self.num_processes = num_processes
        self.num_agents = num_agents
        self.arena_size = arena_size
        self.rng = rng if rng is not None else np.random.RandomState()
        self.goals = np.zeros((num_processes, num_agents, 2), dtype=np.float32)
        self.goal_steps = np.zeros((num_processes, num_agents), dtype=np.float32)
        self.sampled_goals = 0
        self.successes = 0
        self.expired_goals = 0
        self.completed_goal_steps = []

    def initialize(self, env_state=None, envs_info=None):
        self.goals = sample_random_goals(
            env_state,
            self.num_processes,
            self.num_agents,
            self.arena_size,
            rng=self.rng,
            illegal_regions=self._illegal_regions(envs_info),
        )
        self.goal_steps.fill(0.0)
        self.sampled_goals += self.num_processes * self.num_agents
        return self.goals

    def advance_and_resample(self, envs_info, done_agent, env_state=None):
        self.goal_steps += 1.0
        goal_done = np.array(
            [info["goal_done"] for info in envs_info],
            dtype=bool,
        )
        done_agent = np.array(done_agent, dtype=bool)
        reached = goal_done & ~done_agent
        expired = done_agent & ~reached
        resample_mask = reached | expired

        if reached.any():
            self.successes += int(reached.sum())
            self.completed_goal_steps.extend(self.goal_steps[reached].tolist())
        if expired.any():
            self.expired_goals += int(expired.sum())

        illegal_regions = self._illegal_regions(envs_info)
        for proc_idx, agent_idx in np.argwhere(resample_mask):
            self.goals[proc_idx, agent_idx] = _sample_one_goal(
                self.rng,
                self.arena_size,
                illegal_regions,
                max_attempts=128,
            )
            self.goal_steps[proc_idx, agent_idx] = 0.0
            self.sampled_goals += 1

        return self.goals, reached

    def metrics_snapshot(self):
        finished = self.successes + self.expired_goals
        success_rate = self.successes / max(1, finished)
        avg_steps = (
            float(np.mean(self.completed_goal_steps))
            if self.completed_goal_steps else 0.0
        )
        return {
            "goal_success_rate": success_rate,
            "average_steps_per_goal": avg_steps,
            "sampled_goals": self.sampled_goals,
            "completed_goals": self.successes,
            "expired_goals": self.expired_goals,
        }

    def _illegal_regions(self, envs_info):
        if not envs_info:
            return []
        regions = []
        for info in envs_info:
            for obstacle in info.get("obstacles", []):
                if len(obstacle) >= 3:
                    regions.append(tuple(obstacle[:3]))
        return regions
