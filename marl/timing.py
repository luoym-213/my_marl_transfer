import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class TimingProfiler:
    """Low-overhead per-update timing profiler.

    The profiler is intentionally inert unless enabled. CUDA synchronization is
    only performed while profiling so normal training behavior remains unchanged.
    """

    def __init__(self, enabled=False, cuda=False, log_interval=10):
        self.enabled = bool(enabled)
        self.cuda = bool(cuda)
        self.log_interval = max(1, int(log_interval))
        self.current = defaultdict(float)
        self.current_values = defaultdict(list)
        self.current_stats = defaultdict(float)
        self.interval_totals = defaultdict(float)
        self.interval_values = defaultdict(list)
        self.interval_stats = defaultdict(list)
        self.interval_updates = 0
        self.update_start = None

    def reset(self):
        if not self.enabled:
            return
        self.current = defaultdict(float)
        self.current_values = defaultdict(list)
        self.current_stats = defaultdict(float)
        self.update_start = time.perf_counter()

    @contextmanager
    def time(self, name, sync_cuda=True):
        if not self.enabled:
            yield
            return

        self._sync(sync_cuda)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync(sync_cuda)
            self.current[name] += time.perf_counter() - start

    def add(self, name, seconds):
        if self.enabled:
            self.current[name] += float(seconds)

    def add_value(self, name, value):
        if self.enabled:
            self.current_values[name].append(float(value))

    def add_stat(self, name, value):
        if self.enabled:
            self.current_stats[name] += float(value)

    def finish_update(self, update_idx, writer=None):
        if not self.enabled:
            return

        if self.update_start is not None:
            self.current["update_total"] += time.perf_counter() - self.update_start

        for name, seconds in self.current.items():
            self.interval_totals[name] += seconds
            if writer is not None:
                writer.add_scalar("timing/" + name, seconds, update_idx)

        for name, values in self.current_values.items():
            if not values:
                continue
            mean_value = sum(values) / len(values)
            self.interval_values[name].append(mean_value)
            if writer is not None:
                writer.add_scalar("timing/" + name, mean_value, update_idx)

        for name, value in self.current_stats.items():
            self.interval_stats[name].append(value)
            if writer is not None:
                writer.add_scalar("timing/" + name, value, update_idx)

        self.interval_updates += 1
        if self.interval_updates >= self.log_interval:
            self.print_summary(update_idx)
            self.reset_interval()

    def print_summary(self, update_idx):
        if self.interval_updates == 0:
            return

        avg_update = (
            self.interval_totals.get("update_total", 0.0)
            / float(self.interval_updates)
        )
        if avg_update <= 0:
            avg_update = sum(self.interval_totals.values()) / float(
                self.interval_updates
            )

        rows = []
        for name, total in self.interval_totals.items():
            if name == "update_total":
                continue
            avg = total / float(self.interval_updates)
            pct = 100.0 * avg / avg_update if avg_update > 0 else 0.0
            rows.append((avg, pct, total, name))
        rows.sort(reverse=True)

        print(
            "\nTiming profile update {} (last {} updates, avg update {:.4f}s)".format(
                update_idx,
                self.interval_updates,
                avg_update,
            )
        )
        print("{:<34} {:>12} {:>10} {:>12}".format(
            "name", "avg_s", "pct", "total_s"
        ))
        print("-" * 72)
        for avg, pct, total, name in rows:
            print("{:<34} {:>12.4f} {:>9.1f}% {:>12.4f}".format(
                name,
                avg,
                pct,
                total,
            ))

        if self.interval_values:
            print("Timing stats:")
            for name in sorted(self.interval_values):
                vals = self.interval_values[name]
                mean_value = sum(vals) / len(vals)
                print("  {:<32} {:>.4f}".format(name, mean_value))
        if self.interval_stats:
            print("Timing counters:")
            for name in sorted(self.interval_stats):
                vals = self.interval_stats[name]
                mean_value = sum(vals) / len(vals)
                print("  {:<32} {:>.4f}".format(name, mean_value))
        print("")

    def reset_interval(self):
        if not self.enabled:
            return
        self.interval_totals = defaultdict(float)
        self.interval_values = defaultdict(list)
        self.interval_stats = defaultdict(list)
        self.interval_updates = 0

    def _sync(self, sync_cuda=True):
        if self.cuda and sync_cuda:
            torch.cuda.synchronize()
