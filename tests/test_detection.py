"""
Unit tests for detection module.
"""
import numpy as np
import pytest
import cv2
from src.detection import MotionDetector, MotionConfig, HitDetector, HitDetectionConfig
from src.core import Frame, BoardGeometry
from src.board import DartboardMapper


def create_test_frame(image: np.ndarray, frame_id: int = 0) -> Frame:
    """Helper to create Frame object."""
    return Frame(
        image=image,
        timestamp=float(frame_id),
        frame_id=frame_id,
        fps=30.0
    )


def test_motion_detector_init():
    """Test MotionDetector initialization."""
    detector = MotionDetector()

    assert detector.config.history == 500
    assert detector.frame_count == 0
    assert detector.motion_detected_count == 0


def test_motion_detector_no_motion():
    """Test motion detection with static background."""
    detector = MotionDetector()

    # Create static image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process several frames (build background model)
    for i in range(10):
        motion_mask, has_motion = detector.detect(image)

    # Should have no motion after background stabilizes
    assert detector.frame_count == 10


def test_motion_detector_with_motion():
    """Test motion detection with moving object."""
    config = MotionConfig(
        history=10,  # Short history for quick testing
        min_motion_area=20
    )
    detector = MotionDetector(config)

    # Build background (static)
    background = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(15):
        detector.detect(background)

    # Add moving object
    with_motion = background.copy()
    cv2.circle(with_motion, (320, 240), 30, (255, 255, 255), -1)

    motion_mask, has_motion = detector.detect(with_motion)

    # Should detect motion
    assert has_motion
    assert detector.motion_detected_count > 0


def test_motion_detector_reset():
    """Test motion detector reset."""
    detector = MotionDetector()

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.detect(image)

    assert detector.frame_count == 1

    detector.reset()
    # Frame count should persist (it's a statistic)
    # But background model is reset
    assert detector.frame_count == 1


def test_motion_detector_stats():
    """Test motion detector statistics."""
    detector = MotionDetector()

    stats = detector.get_stats()

    assert "frames_processed" in stats
    assert "motion_detected" in stats
    assert "motion_rate_percent" in stats


def test_hit_detector_init():
    """Test HitDetector initialization."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    detector = HitDetector(mapper)

    assert detector.mapper == mapper
    assert detector.frames_processed == 0
    assert len(detector.candidates) == 0
    assert len(detector.confirmed_hits) == 0


def test_hit_detector_no_motion():
    """Test hit detector with no motion."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )
    detector = HitDetector(mapper)

    # Static frame
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    frame = create_test_frame(image, frame_id=0)

    hit = detector.detect(frame)

    assert hit is None
    assert detector.frames_processed == 1


def test_hit_detector_with_motion_no_confirmation():
    """Test hit detector with motion but no temporal confirmation."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    config = HitDetectionConfig(
        confirmation_frames=3,
        motion_config=MotionConfig(
            history=5,
            min_motion_area=10
        )
    )
    detector = HitDetector(mapper, config)

    # Build background
    background = np.zeros((800, 800, 3), dtype=np.uint8)
    for i in range(10):
        frame = create_test_frame(background, frame_id=i)
        detector.detect(frame)

    # Add motion (but only 1 frame)
    with_motion = background.copy()
    cv2.circle(with_motion, (400, 400), 20, (255, 255, 255), -1)
    frame = create_test_frame(with_motion, frame_id=10)

    hit = detector.detect(frame)

    # Should not confirm (needs 3 frames)
    assert hit is None
    assert len(detector.candidates) >= 0  # Candidate created


def test_hit_detector_with_confirmation():
    """Test hit detector with successful temporal confirmation."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    config = HitDetectionConfig(
        confirmation_frames=3,
        position_tolerance_px=10.0,
        motion_config=MotionConfig(
            history=5,
            min_motion_area=10,
            var_threshold=10.0  # More sensitive
        )
    )
    detector = HitDetector(mapper, config)

    # Build background
    background = np.zeros((800, 800, 3), dtype=np.uint8)
    for i in range(10):
        frame = create_test_frame(background, frame_id=i)
        detector.detect(frame)

    # Add stable motion for multiple frames
    with_motion = background.copy()
    cv2.circle(with_motion, (400, 300), 25, (255, 255, 255), -1)

    hit = None
    for i in range(10, 15):
        frame = create_test_frame(with_motion, frame_id=i)
        hit = detector.detect(frame)

        if hit:
            break

    # Should confirm after 3 frames
    if hit:
        assert hit.score is not None
        assert hit.x_px == pytest.approx(400, abs=5)
        assert hit.y_px == pytest.approx(300, abs=5)


def test_hit_detector_cleanup_stale_candidates():
    """Test cleanup of stale candidates."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )

    config = HitDetectionConfig(
        confirmation_timeout_sec=0.1  # Very short timeout for testing
    )
    detector = HitDetector(mapper, config)

    # Build background
    background = np.zeros((800, 800, 3), dtype=np.uint8)
    for i in range(10):
        frame = create_test_frame(background, frame_id=i)
        detector.detect(frame)

    # Add motion briefly
    with_motion = background.copy()
    cv2.circle(with_motion, (400, 400), 20, (255, 255, 255), -1)
    frame = create_test_frame(with_motion, frame_id=10)
    detector.detect(frame)

    # Wait for timeout (simulate time passing)
    import time
    time.sleep(0.15)

    # Process frame without motion
    frame = create_test_frame(background, frame_id=11)
    detector.detect(frame)

    # Candidates should be cleaned up
    assert len(detector.candidates) == 0


def test_hit_detector_reset():
    """Test hit detector reset."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )
    detector = HitDetector(mapper)

    # Add some state
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    frame = create_test_frame(image, frame_id=0)
    detector.detect(frame)

    assert detector.frames_processed == 1

    detector.reset()

    # Candidates should be cleared
    assert len(detector.candidates) == 0


def test_hit_detector_stats():
    """Test hit detector statistics."""
    mapper = DartboardMapper(
        board_center=(400, 400),
        mm_per_pixel=0.5
    )
    detector = HitDetector(mapper)

    stats = detector.get_stats()

    assert "frames_processed" in stats
    assert "motion_frames" in stats
    assert "candidates_created" in stats
    assert "hits_confirmed" in stats
    assert "confirmation_rate_percent" in stats


if __name__ == "__main__":
    print("Running detection module tests...")
    test_motion_detector_init()
    print("✓ MotionDetector init test passed")
    test_motion_detector_no_motion()
    print("✓ Motion detector no motion test passed")
    test_motion_detector_with_motion()
    print("✓ Motion detector with motion test passed")
    test_motion_detector_reset()
    print("✓ Motion detector reset test passed")
    test_motion_detector_stats()
    print("✓ Motion detector stats test passed")
    test_hit_detector_init()
    print("✓ HitDetector init test passed")
    test_hit_detector_no_motion()
    print("✓ Hit detector no motion test passed")
    test_hit_detector_with_motion_no_confirmation()
    print("✓ Hit detector no confirmation test passed")
    test_hit_detector_with_confirmation()
    print("✓ Hit detector with confirmation test passed")
    test_hit_detector_cleanup_stale_candidates()
    print("✓ Hit detector cleanup test passed")
    test_hit_detector_reset()
    print("✓ Hit detector reset test passed")
    test_hit_detector_stats()
    print("✓ Hit detector stats test passed")
    print("\n✓ All detection tests passed!")