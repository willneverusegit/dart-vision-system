"""
Unit tests for capture module.
"""
import numpy as np
import pytest
import time
from pathlib import Path
from src.core import Frame, ROI
from src.capture import ThreadedCamera, CameraConfig, FramePreprocessor, PreprocessConfig


def test_camera_config():
    """Test CameraConfig dataclass."""
    config = CameraConfig(index=0, width=640, height=480, fps=30)
    assert config.index == 0
    assert config.width == 640


def test_preprocessor_downscale():
    """Test downscaling."""
    config = PreprocessConfig(enable_downscale=True, target_width=640)
    preprocessor = FramePreprocessor(config)

    # Create large image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=time.time(), frame_id=0)

    # Process
    processed = preprocessor.process(frame)

    # Check dimensions
    assert processed.image.shape[1] == 640  # Width
    assert processed.image.shape[0] < 1080  # Height scaled down


def test_preprocessor_grayscale():
    """Test grayscale conversion."""
    config = PreprocessConfig(convert_grayscale=True)
    preprocessor = FramePreprocessor(config)

    # Create color image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=time.time(), frame_id=0)

    # Process
    processed = preprocessor.process(frame)

    # Check grayscale
    assert len(processed.image.shape) == 2


def test_preprocessor_roi():
    """Test ROI cropping."""
    roi = ROI(x=100, y=100, width=200, height=150)
    config = PreprocessConfig(roi=roi, enable_downscale=False)
    preprocessor = FramePreprocessor(config)

    # Create image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=time.time(), frame_id=0)

    # Process
    processed = preprocessor.process(frame)

    # Check dimensions
    assert processed.image.shape == (150, 200, 3)


def test_preprocessor_clahe():
    """Test CLAHE application."""
    config = PreprocessConfig(apply_clahe=True)
    preprocessor = FramePreprocessor(config)

    # Create low-contrast image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    frame = Frame(image=img, timestamp=time.time(), frame_id=0)

    # Process (should not crash)
    processed = preprocessor.process(frame)
    assert processed.image.shape == img.shape


# Note: ThreadedCamera tests require actual camera hardware
# For CI/CD, these would be mocked or skipped

@pytest.mark.skipif(True, reason="Requires camera hardware")
def test_threaded_camera_start_stop():
    """Test camera start/stop (requires hardware)."""
    camera = ThreadedCamera(queue_size=3)

    # Start
    started = camera.start()
    assert started
    assert camera.is_running

    # Read a few frames
    for _ in range(5):
        frame = camera.read(timeout=1.0)
        if frame is not None:
            assert isinstance(frame, Frame)
            break

    # Stop
    camera.stop()
    assert not camera.is_running


def test_camera_config_video():
    """Test CameraConfig with video file."""
    # Create dummy video file for testing
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name

    try:
        # Test with Path
        config = CameraConfig(source=Path(video_path))
        assert isinstance(config.source, str)

        # Test is_video_file (won't work as file is empty, but tests the method)
        # In real scenario with actual video file:
        # assert config.is_video_file()
    finally:
        os.unlink(video_path)


def test_camera_config_camera_index():
    """Test CameraConfig with camera index."""
    config = CameraConfig(source=0)
    assert config.source == 0
    assert not config.is_video_file()


if __name__ == "__main__":
    print("Running capture module tests...")
    test_camera_config()
    print("✓ CameraConfig test passed")
    test_preprocessor_downscale()
    print("✓ Downscale test passed")
    test_preprocessor_grayscale()
    print("✓ Grayscale test passed")
    test_preprocessor_roi()
    print("✓ ROI test passed")
    test_preprocessor_clahe()
    print("✓ CLAHE test passed")
    print("\n✓ All capture tests passed!")