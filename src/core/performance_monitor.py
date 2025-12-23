"""
Performance monitoring and profiling.
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0

    # Recent history (sliding window)
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, duration: float) -> None:
        """Update statistics with new timing."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_time = duration
        self.recent_times.append(duration)

    @property
    def avg_time(self) -> float:
        """Average time across all calls."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    @property
    def recent_avg_time(self) -> float:
        """Average of recent calls."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)

    @property
    def avg_fps(self) -> float:
        """Average FPS for this operation."""
        if self.avg_time == 0:
            return 0.0
        return 1.0 / self.avg_time


class PerformanceMonitor:
    """
    Performance monitor for profiling pipeline stages.

    Usage:
        monitor = PerformanceMonitor()

        with monitor.measure("capture"):
            frame = camera.read()

        with monitor.measure("detection"):
            hit = detector.detect(frame)

        stats = monitor.get_report()
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.timings: Dict[str, TimingStats] = {}
        self.enabled = True

        logger.info("PerformanceMonitor initialized")

    def measure(self, name: str):
        """
        Context manager for measuring operation time.

        Args:
            name: Operation name

        Usage:
            with monitor.measure("operation"):
                do_work()
        """
        return TimingContext(self, name)

    def record(self, name: str, duration: float) -> None:
        """
        Record timing manually.

        Args:
            name: Operation name
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        if name not in self.timings:
            self.timings[name] = TimingStats()

        self.timings[name].update(duration)

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get statistics for operation."""
        return self.timings.get(name)

    def get_report(self) -> dict:
        """
        Get complete performance report.

        Returns:
            Dictionary with timing stats for all operations
        """
        report = {}

        for name, stats in self.timings.items():
            report[name] = {
                "avg_time_ms": stats.avg_time * 1000,
                "recent_avg_ms": stats.recent_avg_time * 1000,
                "min_ms": stats.min_time * 1000,
                "max_ms": stats.max_time * 1000,
                "last_ms": stats.last_time * 1000,
                "call_count": stats.call_count,
                "avg_fps": stats.avg_fps,
            }

        return report

    def print_report(self) -> None:
        """Print formatted performance report."""
        print("\n" + "=" * 80)
        print("Performance Report")
        print("=" * 80)

        if not self.timings:
            print("No timing data collected")
            return

        print(f"{'Operation':<20} {'Calls':<10} {'Avg (ms)':<12} {'Recent (ms)':<12} {'FPS':<10}")
        print("-" * 80)

        for name, stats in sorted(self.timings.items()):
            print(
                f"{name:<20} {stats.call_count:<10} "
                f"{stats.avg_time * 1000:<12.2f} {stats.recent_avg_time * 1000:<12.2f} "
                f"{stats.avg_fps:<10.1f}"
            )

        print("=" * 80)

    def reset(self) -> None:
        """Reset all statistics."""
        self.timings.clear()
        logger.info("Performance monitor reset")

    def enable(self) -> None:
        """Enable monitoring."""
        self.enabled = True

    def disable(self) -> None:
        """Disable monitoring (zero overhead)."""
        self.enabled = False


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        """
        Initialize timing context.

        Args:
            monitor: Parent monitor
            name: Operation name
        """
        self.monitor = monitor
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        if self.monitor.enabled:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.monitor.enabled and self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record(self.name, duration)

        return False


class PerformanceProfiler:
    """
    High-level profiler for entire pipeline.

    Tracks:
    - Frame processing time
    - Detection time
    - CPU usage (estimated)
    - Memory usage (estimated)
    - Bottlenecks
    """

    def __init__(self):
        """Initialize profiler."""
        self.monitor = PerformanceMonitor()

        # Pipeline stages
        self.stages = [
            "capture",
            "preprocessing",
            "motion_detection",
            "hit_detection",
            "visualization",
        ]

    def get_bottleneck(self) -> Optional[str]:
        """
        Identify pipeline bottleneck.

        Returns:
            Name of slowest stage
        """
        slowest = None
        slowest_time = 0.0

        for stage in self.stages:
            stats = self.monitor.get_stats(stage)
            if stats and stats.recent_avg_time > slowest_time:
                slowest_time = stats.recent_avg_time
                slowest = stage

        return slowest

    def get_total_time(self) -> float:
        """Get total pipeline time."""
        total = 0.0

        for stage in self.stages:
            stats = self.monitor.get_stats(stage)
            if stats:
                total += stats.recent_avg_time

        return total

    def get_stage_percentages(self) -> dict:
        """Get percentage of time spent in each stage."""
        total = self.get_total_time()

        if total == 0:
            return {}

        percentages = {}
        for stage in self.stages:
            stats = self.monitor.get_stats(stage)
            if stats:
                percentages[stage] = (stats.recent_avg_time / total) * 100

        return percentages

    def print_analysis(self) -> None:
        """Print performance analysis."""
        print("\n" + "=" * 80)
        print("Pipeline Performance Analysis")
        print("=" * 80)

        # Total time
        total = self.get_total_time()
        if total > 0:
            print(f"Total Pipeline Time: {total * 1000:.2f} ms ({1 / total:.1f} FPS)")
        else:
            print("No timing data available")
            return

        print("\nStage Breakdown:")
        print("-" * 80)

        percentages = self.get_stage_percentages()

        for stage in self.stages:
            stats = self.monitor.get_stats(stage)
            if stats:
                pct = percentages.get(stage, 0)
                bar_len = int(pct / 2)
                bar = "█" * bar_len

                print(
                    f"{stage:<20} {stats.recent_avg_time * 1000:>8.2f} ms  "
                    f"{pct:>5.1f}% {bar}"
                )

        # Bottleneck
        bottleneck = self.get_bottleneck()
        if bottleneck:
            print(f"\nBottleneck: {bottleneck}")

        print("=" * 80)