"""
Tests for performance monitoring.
"""
import pytest
import time
from src.core.fps_limiter import FPSLimiter, AdaptiveFPSLimiter
from src.core.performance_monitor import (
    PerformanceMonitor,
    TimingStats,
    PerformanceProfiler,
)


def test_fps_limiter_basic():
    """Test basic FPS limiting."""
    limiter = FPSLimiter(target_fps=10.0)

    assert limiter.target_fps == 10.0
    assert limiter.frame_time == 0.1

    # Process frames
    for _ in range(10):
        limiter.wait()

    # Should be close to target
    assert limiter.actual_fps > 0


def test_fps_limiter_unlimited():
    """Test unlimited FPS."""
    limiter = FPSLimiter(target_fps=0)

    assert limiter.frame_time == 0.0

    # Should not limit
    start = time.time()
    for _ in range(100):
        limiter.wait()
    elapsed = time.time() - start

    # Should be very fast (< 0.1 sec)
    assert elapsed < 0.1


def test_fps_limiter_set_target():
    """Test changing target FPS."""
    limiter = FPSLimiter(target_fps=10.0)

    limiter.set_target_fps(20.0)

    assert limiter.target_fps == 20.0
    assert limiter.frame_time == 0.05


def test_adaptive_fps_modes():
    """Test adaptive FPS mode switching."""
    limiter = AdaptiveFPSLimiter(
        idle_fps=10.0,
        active_fps=25.0,
        cooldown_fps=15.0
    )

    # Initial mode
    assert limiter.target_fps == 10.0

    # Switch to active
    limiter.set_mode("active")
    assert limiter.target_fps == 25.0

    # Switch to cooldown
    limiter.set_mode("cooldown")
    assert limiter.target_fps == 15.0

    # Back to idle
    limiter.set_mode("idle")
    assert limiter.target_fps == 10.0


def test_performance_monitor_measure():
    """Test performance monitoring."""
    monitor = PerformanceMonitor()

    # Measure operation
    with monitor.measure("test_op"):
        time.sleep(0.01)

    stats = monitor.get_stats("test_op")

    assert stats is not None
    assert stats.call_count == 1
    assert stats.avg_time > 0.009  # Should be ~10ms


def test_performance_monitor_multiple():
    """Test multiple operations."""
    monitor = PerformanceMonitor()

    # Measure different operations
    with monitor.measure("op1"):
        time.sleep(0.01)

    with monitor.measure("op2"):
        time.sleep(0.02)

    with monitor.measure("op1"):
        time.sleep(0.01)

    stats1 = monitor.get_stats("op1")
    stats2 = monitor.get_stats("op2")

    assert stats1.call_count == 2
    assert stats2.call_count == 1
    assert stats2.avg_time > stats1.avg_time


def test_timing_stats():
    """Test timing statistics."""
    stats = TimingStats()

    # Add timings
    stats.update(0.01)
    stats.update(0.02)
    stats.update(0.015)

    assert stats.call_count == 3
    assert stats.min_time == 0.01
    assert stats.max_time == 0.02
    assert stats.avg_time == pytest.approx(0.015, rel=0.01)


def test_performance_profiler_bottleneck():
    """Test bottleneck detection."""
    profiler = PerformanceProfiler()

    # Simulate pipeline stages
    with profiler.monitor.measure("capture"):
        time.sleep(0.005)

    with profiler.monitor.measure("preprocessing"):
        time.sleep(0.01)

    with profiler.monitor.measure("motion_detection"):
        time.sleep(0.02)  # Slowest

    with profiler.monitor.measure("hit_detection"):
        time.sleep(0.005)

    bottleneck = profiler.get_bottleneck()

    assert bottleneck == "motion_detection"


def test_performance_monitor_reset():
    """Test monitor reset."""
    monitor = PerformanceMonitor()

    with monitor.measure("test"):
        time.sleep(0.01)

    assert len(monitor.timings) == 1

    monitor.reset()

    assert len(monitor.timings) == 0


def test_performance_monitor_disable():
    """Test disabling monitor."""
    monitor = PerformanceMonitor()

    monitor.disable()

    with monitor.measure("test"):
        time.sleep(0.01)

    # Should not record when disabled
    assert "test" not in monitor.timings