"""Communication degradation primitives for the multi-UAV environments.

The simulator models directed, timestamped packets.  A packet is generated only
when the receiver is inside the communication radius, can be independently
dropped, and otherwise arrives after a capped geometric delay.
"""

from __future__ import division

import heapq

import numpy as np


COMMUNICATION_PROFILES = {
    "mild": {
        "radius": 1.5,
        "mean_delay": 1,
        "max_delay": 2,
        "loss_probability": 0.05,
    },
    "medium": {
        "radius": 0.9,
        "mean_delay": 2,
        "max_delay": 5,
        "loss_probability": 0.15,
    },
    "severe": {
        "radius": 0.6,
        "mean_delay": 4,
        "max_delay": 10,
        "loss_probability": 0.30,
    },
}


def get_communication_profile(level):
    """Return a copy of a paper-defined communication profile."""
    if level not in COMMUNICATION_PROFILES:
        raise ValueError(
            "Unknown communication level {!r}; expected one of {}".format(
                level, sorted(COMMUNICATION_PROFILES)
            )
        )
    return dict(COMMUNICATION_PROFILES[level])


class CommunicationSimulator(object):
    """Generate and deliver causal directed communication packets."""

    def __init__(self, num_agents, level="mild", seed=None, profile=None):
        self.num_agents = int(num_agents)
        self.level = level
        self.profile = (
            dict(profile) if profile is not None
            else get_communication_profile(level)
        )
        self._validate_profile()
        self._rng = np.random.RandomState(seed)
        self.reset()

    def _validate_profile(self):
        required = {
            "radius", "mean_delay", "max_delay", "loss_probability"
        }
        missing = required.difference(self.profile)
        if missing:
            raise ValueError(
                "Communication profile is missing {}".format(sorted(missing))
            )
        if self.profile["radius"] < 0:
            raise ValueError("Communication radius must be non-negative")
        if self.profile["mean_delay"] < 1:
            raise ValueError("Geometric mean delay must be at least one step")
        if self.profile["max_delay"] < 1:
            raise ValueError("Maximum delay must be at least one step")
        loss = self.profile["loss_probability"]
        if not 0.0 <= loss <= 1.0:
            raise ValueError("Loss probability must be in [0, 1]")

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def reset(self):
        self._pending = []
        self._sequence = 0
        self._stats = {
            "attempted": 0,
            "out_of_range": 0,
            "lost": 0,
            "scheduled": 0,
            "delivered": 0,
            "sampled_delay_sum": 0,
        }

    def _sample_delay(self):
        mean_delay = float(self.profile["mean_delay"])
        probability = min(1.0, 1.0 / mean_delay)
        delay = int(self._rng.geometric(probability))
        return min(delay, int(self.profile["max_delay"]))

    def schedule_broadcasts(self, source_step, positions, payloads):
        """Attempt one directed broadcast for every ordered UAV pair.

        ``payloads[sender]`` must be treated as immutable after this call.  The
        same snapshot may be referenced by multiple queued receiver events.
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.num_agents, 2):
            raise ValueError(
                "positions must have shape ({}, 2), got {}".format(
                    self.num_agents, positions.shape
                )
            )
        if len(payloads) != self.num_agents:
            raise ValueError("payload count must equal num_agents")

        radius = float(self.profile["radius"])
        loss_probability = float(self.profile["loss_probability"])
        for sender in range(self.num_agents):
            for receiver in range(self.num_agents):
                if sender == receiver:
                    continue
                self._stats["attempted"] += 1
                distance = float(np.linalg.norm(
                    positions[sender] - positions[receiver]
                ))
                if distance > radius:
                    self._stats["out_of_range"] += 1
                    continue
                if self._rng.random_sample() < loss_probability:
                    self._stats["lost"] += 1
                    continue

                delay = self._sample_delay()
                arrival_step = int(source_step) + delay
                event = (
                    arrival_step,
                    self._sequence,
                    receiver,
                    sender,
                    int(source_step),
                    payloads[sender],
                )
                heapq.heappush(self._pending, event)
                self._sequence += 1
                self._stats["scheduled"] += 1
                self._stats["sampled_delay_sum"] += delay

    def deliver(self, query_step):
        """Return all packets that have arrived by ``query_step`` in order."""
        delivered = []
        while self._pending and self._pending[0][0] <= int(query_step):
            (
                arrival_step,
                _,
                receiver,
                sender,
                source_step,
                payload,
            ) = heapq.heappop(self._pending)
            delivered.append({
                "arrival_step": arrival_step,
                "receiver": receiver,
                "sender": sender,
                "source_step": source_step,
                "payload": payload,
            })
            self._stats["delivered"] += 1
        return delivered

    def get_stats(self):
        stats = dict(self._stats)
        scheduled = stats["scheduled"]
        stats["mean_sampled_delay"] = (
            float(stats["sampled_delay_sum"]) / scheduled
            if scheduled else 0.0
        )
        stats["pending"] = len(self._pending)
        return stats
