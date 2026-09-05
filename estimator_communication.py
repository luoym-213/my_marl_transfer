"""Communication environment exposing causal estimator event streams.

The Chapter-2 policy still receives only ego-local maps and task caches.  This
extension adds the real packets delivered on the current step plus the
receiver's own historical state at each packet's source time.  Predictions are
owned by estimator objects and are never inserted into these histories.
"""

from __future__ import division

import numpy as np

from communication_baseline import (
    CommunicationDegradedEnv,
    EpisodeSeededEnv,
)
import multiagent.scenarios as scenarios


class EstimatorCommunicationEnv(CommunicationDegradedEnv):
    """CommunicationDegradedEnv with an estimator-only delivery interface."""

    def _initialize_local_information(self):
        super(EstimatorCommunicationEnv, self)._initialize_local_information()
        states = self._true_motion_states()
        goals = self._current_goals()
        active = (~np.asarray(self.agents_done, dtype=bool)).astype(np.float32)
        self.local_task_progress = np.zeros((self.n, self.n), dtype=np.float32)
        self.delivery_events = []
        self.delivery_events_by_ego = [[] for _ in range(self.n)]
        self.local_self_histories = [dict() for _ in range(self.n)]
        for ego in range(self.n):
            self.local_self_histories[ego][0] = {
                "motion_state": states[ego].copy(),
                "goal": goals[ego].copy(),
                "option": 0,
                "task_progress": 0.0,
                "active": float(active[ego]),
            }

    def _refresh_own_information(self, tasks):
        super(EstimatorCommunicationEnv, self)._refresh_own_information(tasks)
        states = self._true_motion_states()
        goals = self._current_goals()
        active = (~np.asarray(self.agents_done, dtype=bool)).astype(np.float32)
        progress = np.linalg.norm(goals - states[:, :2], axis=1).astype(
            np.float32
        )
        step = int(self.world.steps)
        for ego in range(self.n):
            self.local_task_progress[ego, ego] = progress[ego]
            self.local_self_histories[ego][step] = {
                "motion_state": states[ego].copy(),
                "goal": goals[ego].copy(),
                "option": int(tasks[ego]),
                "task_progress": float(progress[ego]),
                "active": float(active[ego]),
            }

    def _make_payloads(self, tasks):
        payloads = super(EstimatorCommunicationEnv, self)._make_payloads(tasks)
        for payload in payloads:
            payload["task_progress"] = float(np.linalg.norm(
                np.asarray(payload["goal"], dtype=np.float32)
                - np.asarray(payload["motion_state"], dtype=np.float32)[:2]
            ))
        return payloads

    def _apply_deliveries(self, query_step):
        for delivery_index, event in enumerate(
                self.communication.deliver(query_step)):
            receiver = int(event["receiver"])
            sender = int(event["sender"])
            source_step = int(event["source_step"])
            payload = event["payload"]
            ego_source = self.local_self_histories[receiver].get(source_step)
            if ego_source is None:
                raise RuntimeError(
                    "missing ego {} history at source step {}".format(
                        receiver, source_step
                    )
                )
            public_event = {
                "arrival_step": int(event["arrival_step"]),
                "arrival_time": float(event["arrival_step"] * self.world.dt),
                "sequence": int(delivery_index),
                "receiver": receiver,
                "sender": sender,
                "source_step": source_step,
                "source_time": float(source_step * self.world.dt),
                "motion_state": np.asarray(
                    payload["motion_state"], dtype=np.float32
                ).copy(),
                "option": int(payload["task"]),
                "goal": np.asarray(payload["goal"], dtype=np.float32).copy(),
                "task_progress": float(payload["task_progress"]),
                "active": float(payload["active"]),
                "ego_source_motion_state": np.asarray(
                    ego_source["motion_state"], dtype=np.float32
                ).copy(),
            }
            self.delivery_events.append(public_event)
            self.delivery_events_by_ego[receiver].append(public_event)

            # Local task/map caches and estimators use the same source-time
            # rejection rule, but the raw arrival is still retained above so
            # data tests can audit out-of-order packets.
            if source_step <= self.local_source_steps[receiver, sender]:
                continue
            self.local_agent_states[receiver, sender] = payload["motion_state"]
            self.local_agent_goals[receiver, sender] = payload["goal"]
            self.local_agent_tasks[receiver, sender, 0] = payload["task"]
            self.local_agent_active[receiver, sender] = payload["active"]
            self.local_task_progress[receiver, sender] = payload[
                "task_progress"
            ]
            self.local_source_steps[receiver, sender] = source_step
            self._merge_belief_grid(
                self.local_belief_maps[receiver], payload["belief_grid"]
            )

    def _advance_local_information(self, tasks):
        self.delivery_events = []
        self.delivery_events_by_ego = [[] for _ in range(self.n)]
        super(EstimatorCommunicationEnv, self)._advance_local_information(tasks)

    def _augment_local_info(self, info):
        info = super(EstimatorCommunicationEnv, self)._augment_local_info(info)
        info["local_task_progress"] = self.local_task_progress.copy()
        info["delivery_events"] = [dict(event) for event in self.delivery_events]
        info["delivery_events_by_ego"] = [
            [dict(event) for event in events]
            for events in self.delivery_events_by_ego
        ]
        return info


def make_estimator_communication_env(
        env_id, num_agents, dist_threshold, arena_size, identity_size,
        mask_obs_dist=0.5, communication_level="mild",
        communication_seed=None, trajectory_seed=None):
    """Build the only degraded environment used by estimator experiments."""
    scenario = scenarios.load(env_id + ".py").Scenario(
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
    )
    world = scenario.make_world()
    env = EstimatorCommunicationEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info if hasattr(scenario, "info") else None,
        state_callback=scenario.state,
        discrete_action=True,
        done_callback=scenario.done,
        cam_range=arena_size,
        mask_obs_dist=mask_obs_dist,
        communication_level=communication_level,
        communication_seed=communication_seed,
    )
    if trajectory_seed is None:
        return env
    return EpisodeSeededEnv(
        env,
        trajectory_seed=trajectory_seed,
        communication_seed=communication_seed,
    )
