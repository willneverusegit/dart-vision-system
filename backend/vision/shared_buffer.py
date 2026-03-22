"""Shared memory frame buffer for zero-copy frame passing between capture and processing."""

import logging
from multiprocessing import shared_memory
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """A shared memory buffer for passing video frames between processes.

    Args:
        name: Unique name for the shared memory block.
        width: Frame width in pixels.
        height: Frame height in pixels.
        channels: Number of color channels (default 3 for BGR).
    """

    def __init__(self, name: str, width: int, height: int, channels: int = 3) -> None:
        self._shape = (height, width, channels)
        self._size = height * width * channels
        self._lock = Lock()

        try:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._size)
            logger.info("Created shared memory '%s' (%d bytes)", name, self._size)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            logger.info("Attached to existing shared memory '%s'", name)

    def write(self, frame: np.ndarray) -> None:
        """Write a frame into the shared memory buffer.

        Args:
            frame: Numpy array with shape matching the buffer's expected shape.

        Raises:
            ValueError: If frame shape does not match expected shape.
        """
        if frame.shape != self._shape:
            raise ValueError(
                f"Frame shape {frame.shape} does not match expected shape {self._shape}"
            )
        with self._lock:
            buf = np.ndarray(self._shape, dtype=np.uint8, buffer=self._shm.buf)
            np.copyto(buf, frame)

    def read(self) -> np.ndarray:
        """Read a frame from the shared memory buffer.

        Returns:
            A copy of the frame as a numpy array.
        """
        with self._lock:
            buf = np.ndarray(self._shape, dtype=np.uint8, buffer=self._shm.buf)
            return buf.copy()

    def close(self) -> None:
        """Close the shared memory handle without unlinking."""
        self._shm.close()
        logger.info("Closed shared memory '%s'", self._shm.name)

    def destroy(self) -> None:
        """Close and unlink the shared memory."""
        self._shm.close()
        self._shm.unlink()
        logger.info("Destroyed shared memory '%s'", self._shm.name)

    @property
    def name(self) -> str:
        """Return the shared memory block name."""
        return self._shm.name

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the expected frame shape (height, width, channels)."""
        return self._shape


class FrameBufferPool:
    """Manages multiple named FrameBuffers.

    Example::

        pool = FrameBufferPool()
        pool.create("cam0", 640, 480)
        buf = pool.get("cam0")
        buf.write(frame)
    """

    def __init__(self) -> None:
        self._buffers: dict[str, FrameBuffer] = {}

    def create(self, name: str, width: int, height: int, channels: int = 3) -> FrameBuffer:
        """Create a new FrameBuffer and register it in the pool.

        Args:
            name: Unique buffer name.
            width: Frame width in pixels.
            height: Frame height in pixels.
            channels: Number of color channels.

        Returns:
            The created FrameBuffer.
        """
        buf = FrameBuffer(name=name, width=width, height=height, channels=channels)
        self._buffers[name] = buf
        return buf

    def get(self, name: str) -> FrameBuffer | None:
        """Get a buffer by name.

        Args:
            name: Buffer name.

        Returns:
            The FrameBuffer, or None if not found.
        """
        return self._buffers.get(name)

    def remove(self, name: str) -> None:
        """Destroy and remove a buffer from the pool.

        Args:
            name: Buffer name to remove.
        """
        buf = self._buffers.pop(name, None)
        if buf is not None:
            buf.destroy()

    def destroy_all(self) -> None:
        """Destroy all buffers in the pool."""
        for buf in self._buffers.values():
            buf.destroy()
        self._buffers.clear()
