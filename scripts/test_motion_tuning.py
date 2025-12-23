"""
Test script for motion detection parameter tuning.

Tests different combinations of MOG2 parameters to find optimal settings.
"""
import cv2
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.detection import MotionDetector, MotionConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_motion_config(
        video_path: str,
        config: MotionConfig,
        max_frames: int = 300
):
    """
    Test motion detection with specific config.

    Returns:
        (motion_rate, avg_motion_pixels, fps)
    """
    # Setup camera
    cam_config = CameraConfig(source=video_path, loop_video=False)
    camera = ThreadedCamera(config=cam_config)

    if not camera.start():
        return None

    # Setup detector
    detector = MotionDetector(config)

    # Test
    frame_count = 0
    total_motion_pixels = 0
    import time
    start_time = time.time()

    while frame_count < max_frames:
        frame = camera.read(timeout=1.0)
        if frame is None:
            break

        motion_mask, has_motion = detector.detect(frame.image)

        if has_motion:
            total_motion_pixels += cv2.countNonZero(motion_mask)

        frame_count += 1

    elapsed = time.time() - start_time
    camera.stop()

    # Stats
    stats = detector.get_stats()
    fps = frame_count / elapsed if elapsed > 0 else 0
    avg_motion_pixels = total_motion_pixels / frame_count if frame_count > 0 else 0

    return (
        stats['motion_rate_percent'],
        avg_motion_pixels,
        fps
    )


def main():
    """Run parameter tuning tests."""
    video_path = "videos/a.mp4"

    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return

    print("\n" + "=" * 80)
    print("Motion Detection Parameter Tuning")
    print("=" * 80)

    # Test grid (research-recommended ranges)
    test_configs = [
        # Baseline (current)
        ("Baseline", MotionConfig(
            history=150,
            var_threshold=20.0,
            learning_rate=0.005
        )),

        # Lower variance (more sensitive)
        ("Sensitive", MotionConfig(
            history=150,
            var_threshold=16.0,
            learning_rate=0.005
        )),

        # Higher variance (less noise)
        ("Conservative", MotionConfig(
            history=150,
            var_threshold=24.0,
            learning_rate=0.005
        )),

        # Faster learning
        ("Fast Learn", MotionConfig(
            history=100,
            var_threshold=20.0,
            learning_rate=0.01
        )),

        # With CLAHE
        ("CLAHE", MotionConfig(
            history=150,
            var_threshold=20.0,
            learning_rate=0.005,
            enable_clahe=True
        )),
    ]

    print(f"\nTesting {len(test_configs)} configurations on {video_path}...")
    print(f"{'Config':<15} {'Motion%':<10} {'AvgPixels':<12} {'FPS':<8}")
    print("-" * 80)

    results = []
    for name, config in test_configs:
        result = test_motion_config(video_path, config)

        if result:
            motion_rate, avg_pixels, fps = result
            print(f"{name:<15} {motion_rate:<10.1f} {avg_pixels:<12.1f} {fps:<8.1f}")
            results.append((name, motion_rate, avg_pixels, fps))

    print("=" * 80)
    print("\nRecommendation:")
    print("- Choose config with ~10-30% motion rate for dart games")
    print("- Higher FPS is better (aim for >20 FPS)")
    print("- Adjust based on your specific lighting conditions")
    print()


if __name__ == "__main__":
    main()