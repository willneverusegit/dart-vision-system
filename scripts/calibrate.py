"""
Interactive calibration tool.

Supports:
- Manual 4-point calibration
- ChArUco board calibration (detection only for now)
- Live camera/video preview
- Calibration verification
- Save to config/calib.yaml

Usage:
    # Camera calibration
    python scripts/calibrate.py

    # Video calibration
    python scripts/calibrate.py --video videos/board_view.mp4

    # Specify method
    python scripts/calibrate.py --method manual
    python scripts/calibrate.py --method charuco
"""
import cv2
import sys
import argparse
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig
from src.calibration import ManualCalibrator, CharucoCalibrator
from src.core import Frame, atomic_write_yaml, ensure_config_dir
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive calibration tool for dart vision system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate.py                       # Camera with manual calibration
  python scripts/calibrate.py --method charuco      # Camera with ChArUco
  python scripts/calibrate.py -v videos/board.mp4   # Video with manual calibration
        """
    )

    parser.add_argument(
        "-m", "--method",
        choices=["manual", "charuco"],
        default="manual",
        help="Calibration method (default: manual)"
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "-c", "--camera",
        type=int,
        default=None,
        help="Camera index (default: 0)"
    )
    source_group.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Video file path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="config/calib.yaml",
        help="Output calibration file (default: config/calib.yaml)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save calibration (just preview)"
    )

    return parser.parse_args()


def show_verification(warped_image, calibration_data):
    """
    Show warped image with board overlay for verification.

    Args:
        warped_image: Warped/corrected board image
        calibration_data: Calibration data with parameters
    """
    display = warped_image.copy()

    # Convert to BGR if grayscale
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    # Draw board center
    cx, cy = int(calibration_data.board_center[0]), int(calibration_data.board_center[1])
    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
    cv2.circle(display, (cx, cy), 10, (0, 0, 255), 2)

    # Draw board radius circle
    radius = int(calibration_data.board_radius_px)
    cv2.circle(display, (cx, cy), radius, (0, 255, 0), 2)

    # Draw crosshairs
    cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

    # Add info text
    info_text = [
        f"Method: {calibration_data.method}",
        f"Center: ({cx}, {cy})",
        f"Radius: {radius} px",
        f"Scale: {calibration_data.mm_per_pixel:.4f} mm/px",
    ]

    y_offset = 30
    for text in info_text:
        cv2.putText(
            display,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_offset += 30

    # Show
    cv2.imshow("Calibration Result (Press any key)", display)
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration Result (Press any key)")


def save_calibration(calibration_data, output_path):
    """
    Save calibration to YAML file.

    Args:
        calibration_data: CalibrationData object
        output_path: Output file path
    """
    # Convert to serializable dict
    calib_dict = {
        "calibration": {
            "method": calibration_data.method,
            "timestamp": calibration_data.timestamp,
            "homography_matrix": calibration_data.homography_matrix.tolist(),
            "board_center": list(calibration_data.board_center),
            "mm_per_pixel": float(calibration_data.mm_per_pixel),
            "board_radius_px": float(calibration_data.board_radius_px),
        }
    }

    # Ensure directory exists
    output_path = Path(output_path)
    ensure_config_dir(output_path.parent)

    # Save atomically
    atomic_write_yaml(output_path, calib_dict)
    logger.info(f"Calibration saved to {output_path}")


def capture_frame(camera):
    """
    Capture a single frame from camera.
    Shows live preview until user presses SPACE.

    Args:
        camera: ThreadedCamera instance

    Returns:
        Frame or None
    """
    print("\n" + "=" * 60)
    print("Live Preview - Press SPACE to capture, 'q' to quit")
    print("=" * 60 + "\n")

    while True:
        frame = camera.read(timeout=1.0)

        if frame is None:
            continue

        # Show frame
        display = frame.image.copy()

        # Add instruction overlay
        cv2.putText(
            display,
            "Press SPACE to capture frame for calibration",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capture Frame", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cv2.destroyWindow("Capture Frame")
            return frame
        elif key == ord('q'):
            cv2.destroyWindow("Capture Frame")
            return None


def main():
    """Run interactive calibration."""
    args = parse_args()

    print("=" * 60)
    print("Dart Vision System - Calibration Tool")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")

    # Determine source
    if args.video:
        source = Path(args.video)
        if not source.exists():
            logger.error(f"Video file not found: {source}")
            return
        print(f"Source: Video ({source.name})")
    else:
        source = args.camera if args.camera is not None else 0
        print(f"Source: Camera {source}")

    print("=" * 60 + "\n")

    # Initialize camera
    cam_config = CameraConfig(source=source)
    camera = ThreadedCamera(config=cam_config, queue_size=3)

    if not camera.start():
        logger.error("Failed to start camera/video")
        return

    try:
        # Capture frame for calibration
        frame = capture_frame(camera)

        if frame is None:
            print("Calibration cancelled")
            return

        print("\nFrame captured. Starting calibration...\n")

        # Initialize calibrator
        if args.method == "manual":
            calibrator = ManualCalibrator()
            result = calibrator.calibrate(frame)
        elif args.method == "charuco":
            calibrator = CharucoCalibrator()
            result = calibrator.calibrate(frame, show_detection=True)
        else:
            logger.error(f"Unknown method: {args.method}")
            return

        # Check result
        if not result.success:
            print(f"\n❌ Calibration failed: {result.message}")
            return

        print(f"\n✓ Calibration successful!")
        print(f"  Method: {result.data.method}")
        print(f"  Scale: {result.data.mm_per_pixel:.4f} mm/px")
        print(f"  Center: {result.data.board_center}")
        print(f"  Radius: {result.data.board_radius_px:.1f} px")

        # Show verification
        if result.warped_image is not None:
            print("\nShowing calibration result...")
            show_verification(result.warped_image, result.data)

        # Save calibration
        if not args.no_save:
            save_calibration(result.data, args.output)
            print(f"\n✓ Calibration saved to {args.output}")
        else:
            print("\n(Calibration not saved - --no-save flag used)")

        print("\n" + "=" * 60)
        print("Calibration complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        logger.error(f"Calibration error: {e}", exc_info=True)

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()