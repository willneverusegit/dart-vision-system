"""Tests for the shared memory frame buffer."""

from multiprocessing import shared_memory

import numpy as np
import pytest

from backend.vision.shared_buffer import FrameBuffer, FrameBufferPool


@pytest.fixture
def frame_480p() -> np.ndarray:
    """A random 480x640 BGR test frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def buffer() -> FrameBuffer:
    """Create a FrameBuffer and ensure cleanup."""
    buf = FrameBuffer(name="test_buf", width=640, height=480)
    yield buf
    try:
        buf.destroy()
    except FileNotFoundError:
        pass


def test_write_read_frame(buffer: FrameBuffer, frame_480p: np.ndarray) -> None:
    """Writing then reading a frame returns identical data."""
    buffer.write(frame_480p)
    result = buffer.read()
    assert np.array_equal(result, frame_480p)


def test_shape_mismatch_raises(buffer: FrameBuffer) -> None:
    """Writing a frame with the wrong shape raises ValueError."""
    wrong = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match expected shape"):
        buffer.write(wrong)


def test_read_returns_copy(buffer: FrameBuffer, frame_480p: np.ndarray) -> None:
    """Modifying the returned array does not affect the buffer."""
    buffer.write(frame_480p)
    result = buffer.read()
    result[:] = 0
    second = buffer.read()
    assert np.array_equal(second, frame_480p)


def test_multiple_buffers() -> None:
    """FrameBufferPool manages independent buffers."""
    pool = FrameBufferPool()
    try:
        pool.create("pool_a", 320, 240)
        pool.create("pool_b", 160, 120)

        frame_a = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)

        pool.get("pool_a").write(frame_a)
        pool.get("pool_b").write(frame_b)

        assert np.array_equal(pool.get("pool_a").read(), frame_a)
        assert np.array_equal(pool.get("pool_b").read(), frame_b)
    finally:
        pool.destroy_all()


def test_destroy_cleans_up() -> None:
    """After destroy, the shared memory block no longer exists."""
    buf = FrameBuffer(name="test_destroy", width=64, height=64)
    shm_name = buf.name
    buf.destroy()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm_name, create=False)
