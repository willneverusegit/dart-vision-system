"""
Tests for detection state machine.
"""
import pytest
import time
from src.detection.detection_state import (
    DetectionState,
    DetectionStateMachine,
    StateConfig,
)


def test_initial_state():
    """Test initial state is IDLE."""
    sm = DetectionStateMachine()
    assert sm.state == DetectionState.IDLE


def test_idle_to_watching():
    """Test transition from IDLE to WATCHING on motion."""
    sm = DetectionStateMachine()

    # No motion - stay idle
    sm.update(has_motion=False)
    assert sm.state == DetectionState.IDLE

    # Motion detected - transition to watching
    sm.update(has_motion=True)
    assert sm.state == DetectionState.WATCHING


def test_watching_to_confirming():
    """Test transition from WATCHING to CONFIRMING."""
    config = StateConfig(watching_duration_sec=0.1)
    sm = DetectionStateMachine(config)

    # Enter watching
    sm.update(has_motion=True)
    assert sm.state == DetectionState.WATCHING

    # Motion stops - transition to confirming
    sm.update(has_motion=False)
    assert sm.state == DetectionState.CONFIRMING


def test_watching_timeout():
    """Test WATCHING timeout."""
    config = StateConfig(watching_duration_sec=0.05)
    sm = DetectionStateMachine(config)

    # Enter watching
    sm.update(has_motion=True)
    assert sm.state == DetectionState.WATCHING

    # Wait for timeout
    time.sleep(0.1)
    sm.update(has_motion=True)

    assert sm.state == DetectionState.CONFIRMING


def test_hand_filtering():
    """Test large blob (hand) detection."""
    config = StateConfig(large_blob_threshold_px=500)
    sm = DetectionStateMachine(config)

    # Enter watching
    sm.update(has_motion=True, motion_pixels=100)
    assert sm.state == DetectionState.WATCHING

    # Large blob (hand) - stay in watching
    sm.update(has_motion=True, motion_pixels=1000)
    assert sm.state == DetectionState.WATCHING
    assert sm.should_ignore_motion()


def test_confirming_to_cooldown():
    """Test hit confirmation."""
    sm = DetectionStateMachine()

    # Get to confirming state
    sm.update(has_motion=True)  # → WATCHING
    sm.update(has_motion=False)  # → CONFIRMING
    assert sm.state == DetectionState.CONFIRMING

    # Confirm hit
    sm.update(has_motion=False, confirmed_hit=(100.0, 100.0))
    assert sm.state == DetectionState.COOLDOWN
    assert sm.last_hit_position == (100.0, 100.0)


def test_confirming_timeout():
    """Test CONFIRMING timeout."""
    config = StateConfig(confirming_duration_sec=0.05)
    sm = DetectionStateMachine(config)

    # Get to confirming
    sm.update(has_motion=True)
    sm.update(has_motion=False)
    assert sm.state == DetectionState.CONFIRMING

    # Wait for timeout
    time.sleep(0.1)
    sm.update(has_motion=False)

    assert sm.state == DetectionState.IDLE


def test_cooldown_zone():
    """Test cooldown zone detection."""
    config = StateConfig(cooldown_radius_px=50.0)
    sm = DetectionStateMachine(config)

    # Get to cooldown
    sm.update(has_motion=True)
    sm.update(has_motion=False)
    sm.update(has_motion=False, confirmed_hit=(100.0, 100.0))
    assert sm.state == DetectionState.COOLDOWN

    # Check cooldown zone
    assert sm.is_in_cooldown_zone(105.0, 105.0)  # Within radius
    assert not sm.is_in_cooldown_zone(200.0, 200.0)  # Outside radius


def test_cooldown_to_idle():
    """Test COOLDOWN timeout."""
    config = StateConfig(cooldown_duration_sec=0.05)
    sm = DetectionStateMachine(config)

    # Get to cooldown
    sm.update(has_motion=True)
    sm.update(has_motion=False)
    sm.update(has_motion=False, confirmed_hit=(100.0, 100.0))
    assert sm.state == DetectionState.COOLDOWN

    # Wait for cooldown
    time.sleep(0.1)
    sm.update(has_motion=False)

    assert sm.state == DetectionState.IDLE


def test_state_transitions_count():
    """Test transition counting."""
    sm = DetectionStateMachine()

    initial = sm.state_transitions

    sm.update(has_motion=True)  # → WATCHING
    sm.update(has_motion=False)  # → CONFIRMING

    assert sm.state_transitions == initial + 2


def test_process_detection_flag():
    """Test should_process_detection flag."""
    sm = DetectionStateMachine()

    # Not in confirming - don't process
    assert not sm.should_process_detection()

    # Get to confirming
    sm.update(has_motion=True)
    sm.update(has_motion=False)

    # In confirming - process
    assert sm.should_process_detection()


def test_reset():
    """Test state machine reset."""
    sm = DetectionStateMachine()

    # Change state
    sm.update(has_motion=True)
    sm.update(has_motion=False)
    sm.update(has_motion=False, confirmed_hit=(100.0, 100.0))

    # Reset
    sm.reset()

    assert sm.state == DetectionState.IDLE
    assert sm.last_hit_position is None
    assert sm.last_hit_time is None