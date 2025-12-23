"""
Live demo of automatic dart hit detection.

Integrates all modules:
- Camera capture
- Calibration loading
- Motion detection
- Hit recognition
- Scoring and visualization

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py -v videos/game.mp4
    python scripts/live_demo.py -c 1 --calib config/calib.yaml
"""
import cv2
import sys
import argparse
from pathlib import Path
import numpy as np
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.core import CalibrationData, load_yaml, BoardGeometry
from src.board import DartboardMapper, BoardVisualizer
from src.detection import HitDetector, HitDetectionConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveDemo:
    """Live demonstration of automatic dart hit detection."""

    def __init__(
            self,
            camera: ThreadedCamera,
            calibration: CalibrationData,
            show_debug: bool = False
    ):
        """
        Initialize live demo.

        Args:
            camera: Threaded camera instance
            calibration: Loaded calibration data
            show_debug: Show debug visualizations
        """
        self.camera = camera
        self.calib = calibration
        self.show_debug = show_debug

        # Initialize mapper and visualizer
        self.mapper = DartboardMapper(
            board_geometry=BoardGeometry(),
            board_center=calibration.board_center,
            mm_per_pixel=calibration.mm_per_pixel
        )

        self.visualizer = BoardVisualizer(self.mapper, opacity=0.3)

        # Initialize hit detector
        detection_config = HitDetectionConfig()
        self.hit_detector = HitDetector(self.mapper, config=detection_config)

        # Game state
        self.hits = []
        self.total_score = 0

        # UI state
        self.show_overlay = True
        self.show_motion = False
        self.paused = False

        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        logger.info("LiveDemo initialized")

    def run(self):
        """Run live demo."""
        print("\n" + "=" * 60)
        print("Dart Vision System - Live Demo")
        print("=" * 60)
        print("Controls:")
        print("  'o': Toggle board overlay")
        print("  'm': Toggle motion mask view")
        print("  'r': Reset detection (clear hits)")
        print("  'p': Pause/Resume")
        print("  's': Show statistics")
        print("  'q': Quit")
        print("=" * 60 + "\n")

        # Windows
        cv2.namedWindow("Live Demo")
        if self.show_debug:
            cv2.namedWindow("Motion Mask")

        try:
            last_motion_mask = None

            while True:
                if not self.paused:
                    # Read frame
                    frame = self.camera.read(timeout=1.0)

                    if frame is None:
                        continue

                    # Update FPS
                    self._update_fps()

                    # Warp frame to calibrated view
                    warped = cv2.warpPerspective(
                        frame.image,
                        self.calib.homography_matrix,
                        (800, 800)
                    )

                    # Create Frame object for detection
                    detection_frame = type('Frame', (), {
                        'image': warped,
                        'frame_id': frame.frame_id,
                        'timestamp': frame.timestamp,
                        'fps': frame.fps
                    })()

                    # Detect hit
                    hit = self.hit_detector.detect(detection_frame)

                    if hit:
                        self.hits.append(hit)
                        self.total_score += hit.score
                        logger.info(f"New hit: {hit.score} points (Total: {self.total_score})")

                    # Get motion mask for visualization
                    if self.show_motion or self.show_debug:
                        # Access internal motion detector for mask
                        motion_mask, _ = self.hit_detector.motion_detector.detect(warped)
                        last_motion_mask = motion_mask

                    # Visualize
                    display = self._create_display(warped, last_motion_mask)

                    cv2.imshow("Live Demo", display)

                    if self.show_debug and last_motion_mask is not None:
                        cv2.imshow("Motion Mask", last_motion_mask)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('o'):
                    self.show_overlay = not self.show_overlay
                    logger.info(f"Overlay: {'ON' if self.show_overlay else 'OFF'}")
                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                    logger.info(f"Motion view: {'ON' if self.show_motion else 'OFF'}")
                elif key == ord('r'):
                    self._reset()
                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"{'PAUSED' if self.paused else 'RESUMED'}")
                elif key == ord('s'):
                    self._print_stats()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            cv2.destroyAllWindows()
            self._print_summary()

    def _create_display(
            self,
            warped: np.ndarray,
            motion_mask: np.ndarray = None
    ) -> np.ndarray:
        """Create display image with overlays."""
        display = warped.copy()

        # Show motion mask overlay (semi-transparent)
        if self.show_motion and motion_mask is not None:
            # ← NEU: Draw contours on motion mask for debugging
            motion_debug = self._draw_motion_debug(warped, motion_mask)
            display = motion_debug

        # Draw board overlay
        if self.show_overlay:
            display = self.visualizer.draw_board_overlay(display)

        # Draw all hits
        if self.hits:
            display = self.visualizer.draw_hits(display, self.hits[-10:], show_scores=True)

        # ← NEU: Draw active candidates
        display = self._draw_candidates(display)

        # Draw HUD
        display = self._draw_hud(display)

        return display

    def _draw_motion_debug(
            self,
            image: np.ndarray,
            motion_mask: np.ndarray
    ) -> np.ndarray:
        """Draw motion detection debug visualization."""
        debug = image.copy()

        # Find contours in motion mask
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw all contours in yellow
        cv2.drawContours(debug, contours, -1, (0, 255, 255), 2)

        # Draw filtered contours in green
        config = self.hit_detector.config
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.min_contour_area or area > config.max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = w / h
            if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
                continue

            # Valid contour - draw in green
            cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)

            # Draw bounding box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Draw area text
            cv2.putText(
                debug,
                f"A={area:.0f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        return debug

    def _draw_candidates(self, image: np.ndarray) -> np.ndarray:
        """Draw active hit candidates."""
        result = image.copy()

        for candidate in self.hit_detector.candidates:
            x, y = candidate.position
            x, y = int(x), int(y)

            # Draw candidate position
            color = (255, 255, 0)  # Yellow for unconfirmed
            cv2.circle(result, (x, y), 5, color, 2)

            # Draw confirmation progress
            progress_text = f"{candidate.confirmation_count}/{self.hit_detector.config.confirmation_frames}"
            cv2.putText(
                result,
                progress_text,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        return result

    def _draw_hud(self, image: np.ndarray) -> np.ndarray:
        """Draw heads-up display."""
        # Info panel
        info = {
            "FPS": f"{self.current_fps:.1f}",
            "Hits": len(self.hits),
            "Score": self.total_score,
            "Candidates": len(self.hit_detector.candidates),
        }

        if self.paused:
            info["Status"] = "PAUSED"

        result = self.visualizer.draw_info_panel(image, info, position=(10, 30))

        # Instructions at bottom
        instructions = [
            "Press 'o' for overlay",
            "'m' for motion",
            "'r' to reset",
            "'q' to quit"
        ]

        y_offset = image.shape[0] - 20
        for i, text in enumerate(instructions):
            cv2.putText(
                result,
                text,
                (10, y_offset - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        return result

    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1

        if self.fps_frame_count >= 30:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_frame_count / elapsed

            # Reset
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def _reset(self):
        """Reset detection and game state."""
        self.hits.clear()
        self.total_score = 0
        self.hit_detector.reset()
        logger.info("Reset complete")

    def _print_stats(self):
        """Print detection statistics."""
        stats = self.hit_detector.get_stats()

        print("\n" + "=" * 60)
        print("Detection Statistics")
        print("=" * 60)
        print(f"Frames processed:      {stats['frames_processed']}")
        print(f"Motion frames:         {stats['motion_frames']} ({stats['motion_rate_percent']:.1f}%)")
        print(f"Candidates created:    {stats['candidates_created']}")
        print(f"Hits confirmed:        {stats['hits_confirmed']}")
        print(f"Confirmation rate:     {stats['confirmation_rate_percent']:.1f}%")
        print(f"Active candidates:     {stats['active_candidates']}")
        print(f"Current FPS:           {self.current_fps:.1f}")
        print("=" * 60 + "\n")

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total hits:            {len(self.hits)}")
        print(f"Total score:           {self.total_score}")

        if self.hits:
            print(f"Average per hit:       {self.total_score / len(self.hits):.1f}")
            print(f"\nLast 5 hits:")
            for i, hit in enumerate(self.hits[-5:], 1):
                multiplier_str = {1: "", 2: "D", 3: "T", 25: "SB", 50: "DB"}
                mult_label = multiplier_str.get(hit.multiplier, str(hit.multiplier))
                sector_label = str(hit.sector) if hit.sector else ""
                print(f"  {i}. {mult_label}{sector_label} = {hit.score} points")

        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live demo of automatic dart hit detection"
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug windows"
    )

    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Don't loop video (default: loop)"
    )

    return parser.parse_args()


def main():
    """Run live demo."""
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

    # Determine source
    if args.video:
        source = args.video
        loop = not args.no_loop
    elif args.camera is not None:
        source = args.camera
        loop = False
    else:
        source = 0  # Default camera
        loop = False

    # Initialize camera
    cam_config = CameraConfig(source=source, loop_video=loop)
    camera = ThreadedCamera(config=cam_config, queue_size=3)

    if not camera.start():
        logger.error("Failed to start camera/video")
        return

    try:
        # Run demo
        demo = LiveDemo(camera, calib_data, show_debug=args.debug)
        demo.run()

    finally:
        camera.stop()


if __name__ == "__main__":
    main()