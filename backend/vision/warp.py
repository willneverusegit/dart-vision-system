"""Dedicated warp module for top-down board view transformation."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WarpEngine:
    """Holds a homography matrix and provides frame warping methods.

    Attributes:
        _homography: The 3x3 perspective transform matrix.
        _output_size: Side length of the square output image.
    """

    def __init__(self) -> None:
        self._homography: np.ndarray | None = None
        self._output_size: int = 500

    def set_homography(self, homography: np.ndarray, output_size: int = 500) -> None:
        """Update the homography matrix and output size.

        Args:
            homography: A 3x3 homography matrix.
            output_size: Side length of the square output in pixels.

        Raises:
            ValueError: If homography is not a 3x3 matrix.
        """
        arr = np.asarray(homography, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {arr.shape}")
        self._homography = arr
        self._output_size = output_size
        logger.info("Homography configured with output_size=%d", output_size)

    def is_configured(self) -> bool:
        """Return True if a homography has been set."""
        return self._homography is not None

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp a BGR frame to top-down view using the stored homography.

        Args:
            frame: Input BGR image as numpy array.

        Returns:
            Warped square image.

        Raises:
            RuntimeError: If no homography is configured.
        """
        if self._homography is None:
            raise RuntimeError("WarpEngine not configured — call set_homography first")
        return cv2.warpPerspective(frame, self._homography, (self._output_size, self._output_size))

    def warp_frame_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """Warp a frame and encode the result as JPEG bytes.

        Args:
            frame: Input BGR image.
            quality: JPEG quality (0-100).

        Returns:
            JPEG-encoded bytes of the warped image.

        Raises:
            RuntimeError: If no homography is configured or encoding fails.
        """
        warped = self.warp_frame(frame)
        ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return bytes(buf)


warp_engine = WarpEngine()
