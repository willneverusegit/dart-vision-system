"""
Interactive scoring test tool.

Click on warped dartboard image to test scoring accuracy.

Usage:
    python scripts/test_scoring.py
    python scripts/test_scoring.py --calib config/calib.yaml
    python scripts/test_scoring.py --video videos/board.mp4
"""
import cv2
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.core import CalibrationData, load_yaml, BoardGeometry
from src.board import DartboardMapper, BoardVisualizer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScoringTester:
    """Interactive scoring tester."""

    def __init__(
            self,
            calibration_data: CalibrationData,
            source_image: np.ndarray
    ):
        """
        Initialize scoring tester.

        Args:
            calibration_data: Loaded calibration
            source_image: Source image to warp and test on
        """
        self.calib = calibration_data
        self.source_image = source_image

        # Warp image
        self.warped_size = 800
        H = calibration_data.homography_matrix
        self.warped_image = cv2.warpPerspective(
            source_image,
            H,
            (self.warped_size, self.warped_size)
        )

        # Initialize mapper and visualizer
        self.mapper = DartboardMapper(
            board_geometry=BoardGeometry(),
            board_center=calibration_data.board_center,
            mm_per_pixel=calibration_data.mm_per_pixel
        )

        self.visualizer = BoardVisualizer(self.mapper, opacity=0.3)

        # Display image with overlay
        self.display_image = self.visualizer.draw_board_overlay(
            self.warped_image.copy()
        )

        # Hits tracking
        self.hits = []

        # Window
        self.window_name = "Scoring Test - Click to test"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculate score
            hit = self.mapper.pixel_to_score(float(x), float(y))
            self.hits.append(hit)

            # Update display
            self._update_display()

            # Log
            score_str = f"{hit.multiplier}x{hit.sector}" if hit.sector else str(hit.score)
            logger.info(f"Click at ({x}, {y}) -> {score_str} = {hit.score} points")

    def _update_display(self):
        """Update display with hits."""
        # Start with base overlay
        self.display_image = self.visualizer.draw_board_overlay(
            self.warped_image.copy()
        )

        # Draw all hits
        self.display_image = self.visualizer.draw_hits(
            self.display_image,
            self.hits,
            show_scores=True
        )

        # Info panel
        total_score = sum(h.score for h in self.hits if h.score)
        info = {
            "Hits": len(self.hits),
            "Total": total_score,
            "Last": self.hits[-1].score if self.hits else 0
        }

        self.display_image = self.visualizer.draw_info_panel(
            self.display_image,
            info
        )

    def run(self):
        """Run interactive test."""
        print("\n" + "=" * 60)
        print("Interactive Scoring Test")
        print("=" * 60)
        print("Controls:")
        print("  - Left Click: Test score at position")
        print("  - 'r': Reset hits")
        print("  - 'q': Quit")
        print("=" * 60 + "\n")

        while True:
            cv2.imshow(self.window_name, self.display_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.hits = []
                self._update_display()
                logger.info("Hits reset")

        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total hits: {len(self.hits)}")
        if self.hits:
            total_score = sum(h.score for h in self.hits if h.score)
            print(f"  Total score: {total_score}")
            print(f"  Average: {total_score / len(self.hits):.1f}")
        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive scoring test tool"
    )

    parser.add_argument(
        "--calib",
        type=str,
        default="config/calib.yaml",
        help="Calibration file (default: config/calib.yaml)"
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "-c", "--camera",
        type=int,
        default=None,
        help="Camera index"
    )
    source_group.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Video file path"
    )

    return parser.parse_args()


def main():
    """Run scoring test."""
    args = parse_args()

    # Load calibration
    calib_path = Path(args.calib)
    if not calib_path.exists():
        logger.error(f"Calibration file not found: {calib_path}")
        logger.info("Run 'python scripts/calibrate.py' first")
        return

    try:
        calib_dict = load_yaml(calib_path)
        calib_data = CalibrationData(
            homography_matrix=np.array(calib_dict["calibration"]["homography_matrix"]),
            board_center=tuple(calib_dict["calibration"]["board_center"]),
            mm_per_pixel=calib_dict["calibration"]["mm_per_pixel"],
            board_radius_px=calib_dict["calibration"]["board_radius_px"],
            method=calib_dict["calibration"]["method"],
            timestamp=calib_dict["calibration"].get("timestamp")
        )
        logger.info(f"Calibration loaded: {calib_data.method}, {calib_data.mm_per_pixel:.4f} mm/px")
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        return

    # Get source image
    if args.video or args.camera is not None:
        # Capture from camera/video
        source = args.video if args.video else (args.camera if args.camera is not None else 0)

        cam_config = CameraConfig(source=source, loop_video=False)
        camera = ThreadedCamera(config=cam_config)

        if not camera.start():
            logger.error("Failed to start capture")
            return

        logger.info("Capturing frame... Press SPACE")

        while True:
            frame = camera.read(timeout=1.0)
            if frame is None:
                continue

            cv2.imshow("Capture (press SPACE)", frame.image)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                source_image = frame.image
                break

        camera.stop()
        cv2.destroyAllWindows()
    else:
        # Use default test image (black with white circle)
        source_image = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.circle(source_image, (400, 400), 300, (255, 255, 255), 2)
        logger.info("Using synthetic test image")

    # Run tester
    tester = ScoringTester(calib_data, source_image)
    tester.run()


if __name__ == "__main__":
    main()