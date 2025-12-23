"""
Benchmark script for performance testing.

Tests different configurations and reports metrics.
"""
import cv2
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.core import CalibrationData, load_yaml, PerformanceProfiler
from src.board import DartboardMapper, BoardVisualizer, BoardGeometry
from src.detection import HitDetector, HitDetectionConfig
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_configuration(
        video_path: str,
        calib_data: CalibrationData,
        config: HitDetectionConfig,
        max_frames: int = 300,
        config_name: str = "Default"
) -> dict:
    """
    Benchmark a specific configuration.

    Returns:
        Performance metrics
    """
    logger.info(f"Benchmarking: {config_name}")

    # Setup
    cam_config = CameraConfig(source=video_path, loop_video=False)
    camera = ThreadedCamera(config=cam_config)

    if not camera.start():
        return None

    mapper = DartboardMapper(
        board_geometry=BoardGeometry(),
        board_center=calib_data.board_center,
        mm_per_pixel=calib_data.mm_per_pixel
    )

    detector = HitDetector(mapper, config=config)
    profiler = PerformanceProfiler()

    # Benchmark
    frame_count = 0
    hits_detected = 0
    start_time = time.time()

    while frame_count < max_frames:
        # Capture
        with profiler.monitor.measure("capture"):
            frame = camera.read(timeout=1.0)

        if frame is None:
            break

        # Warp
        with profiler.monitor.measure("preprocessing"):
            warped = cv2.warpPerspective(
                frame.image,
                calib_data.homography_matrix,
                (800, 800)
            )

        # Detect
        with profiler.monitor.measure("detection"):
            hit = detector.detect(
                type('Frame', (), {
                    'image': warped,
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'fps': frame.fps
                })()
            )

        if hit:
            hits_detected += 1

        frame_count += 1

    elapsed = time.time() - start_time
    camera.stop()

    # Results
    total_time = profiler.get_total_time()
    pipeline_fps = 1.0 / total_time if total_time > 0 else 0

    return {
        "config_name": config_name,
        "frames_processed": frame_count,
        "elapsed_sec": elapsed,
        "realtime_fps": frame_count / elapsed,
        "pipeline_fps": pipeline_fps,
        "hits_detected": hits_detected,
        "profiler": profiler,
    }


def main():
    """Run performance benchmarks."""
    video_path = "videos/a.mp4"
    calib_path = Path("config/calib.yaml")

    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return

    if not calib_path.exists():
        logger.error(f"Calibration not found: {calib_path}")
        return

    # Load calibration
    calib_dict = load_yaml(calib_path)
    calib_data = CalibrationData(
        homography_matrix=np.array(calib_dict["calibration"]["homography_matrix"]),
        board_center=tuple(calib_dict["calibration"]["board_center"]),
        mm_per_pixel=calib_dict["calibration"]["mm_per_pixel"],
        board_radius_px=calib_dict["calibration"]["board_radius_px"],
        method=calib_dict["calibration"]["method"],
    )

    print("\n" + "=" * 80)
    print("Performance Benchmark Suite")
    print("=" * 80)

    # Test configurations
    configs = [
        ("Baseline", HitDetectionConfig()),

        ("Low CPU", HitDetectionConfig(
            confirmation_frames=3,
            enable_subpixel=False,
        )),

        ("High Accuracy", HitDetectionConfig(
            confirmation_frames=7,
            enable_subpixel=True,
        )),

        ("Fast Motion", HitDetectionConfig(
            motion_config__history=100,
            motion_config__var_threshold=24.0,
        )),
    ]

    results = []

    for name, config in configs:
        result = benchmark_configuration(
            video_path,
            calib_data,
            config,
            max_frames=300,
            config_name=name
        )

        if result:
            results.append(result)

    # Print comparison
    print("\n" + "=" * 80)
    print("Configuration Comparison")
    print("=" * 80)
    print(f"{'Config':<20} {'RT FPS':<10} {'Pipe FPS':<10} {'Hits':<8}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['config_name']:<20} "
            f"{result['realtime_fps']:<10.1f} "
            f"{result['pipeline_fps']:<10.1f} "
            f"{result['hits_detected']:<8}"
        )

    print("=" * 80)

    # Detailed analysis for best config
    best = max(results, key=lambda r: r['pipeline_fps'])

    print(f"\nBest Configuration: {best['config_name']}")
    best['profiler'].print_analysis()


if __name__ == "__main__":
    main()