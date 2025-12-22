"""
Live camera/video test script.
Tests threaded capture and preprocessing with real-time display.

Usage:
    # Camera mode (default)
    python scripts/test_camera.py

    # Video mode
    python scripts/test_camera.py --video videos/test.mp4
    python scripts/test_camera.py -v videos/test.mp4

    # Specify camera index
    python scripts/test_camera.py --camera 1

Controls:
    - 'q': Quit
    - 'g': Toggle grayscale
    - 'c': Toggle CLAHE
    - 's': Save current frame
    - 'SPACE': Pause/Resume (video only)
"""
import cv2
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import ThreadedCamera, CameraConfig, FramePreprocessor, PreprocessConfig
from src.core import ROI
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
        description="Test threaded camera/video capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_camera.py                    # Use camera 0
  python scripts/test_camera.py --camera 1         # Use camera 1
  python scripts/test_camera.py -v videos/test.mp4 # Use video file
        """
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
        "--no-loop",
        action="store_true",
        help="Don't loop video (exit when finished)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera resolution width (default: 1280)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera resolution height (default: 720)"
    )

    return parser.parse_args()


def main():
    """Run live camera/video test."""
    args = parse_args()

    # Determine source
    if args.video:
        source = Path(args.video)
        if not source.exists():
            logger.error(f"Video file not found: {source}")
            return
        source_str = f"Video: {source.name}"
    else:
        source = args.camera if args.camera is not None else 0
        source_str = f"Camera {source}"

    print("=" * 60)
    print("Threaded Capture Test")
    print(f"Source: {source_str}")
    print("=" * 60)
    print("Controls:")
    print("  'q'     - Quit")
    print("  'g'     - Toggle grayscale")
    print("  'c'     - Toggle CLAHE")
    print("  's'     - Save frame")
    if args.video:
        print("  'SPACE' - Pause/Resume")
    print("=" * 60)

    # Configure camera/video
    cam_config = CameraConfig(
        source=source,
        width=args.width,
        height=args.height,
        fps=30,
        loop_video=not args.no_loop
    )

    # Configure preprocessor
    preproc_config = PreprocessConfig(
        enable_downscale=True,
        target_width=1280,
        convert_grayscale=False,
        apply_clahe=False
    )

    # Initialize
    camera = ThreadedCamera(config=cam_config, queue_size=3)
    preprocessor = FramePreprocessor(config=preproc_config)

    # Start camera
    if not camera.start():
        logger.error("Failed to start capture")
        return

    print(f"\nCapture properties: {camera.get_capture_properties()}")
    print("Capture started. Press 'q' to quit.\n")

    frame_count = 0
    paused = False

    try:
        while True:
            # Handle pause (video only)
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    paused = False
                    print("Resumed")
                elif key == ord('q'):
                    break
                continue

            # Read frame
            frame = camera.read(timeout=1.0)

            if frame is None:
                if camera.is_video and not cam_config.loop_video:
                    logger.info("Video finished")
                    break
                logger.warning("No frame received")
                continue

            # Preprocess
            processed = preprocessor.process(frame)

            # Display info overlay
            display_img = processed.image.copy()

            # Convert to BGR if grayscale (for colored text overlay)
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            # Build info text
            info_text = [
                f"Source: {source_str}",
                f"FPS: {camera.fps:.1f}",
                f"Frame: {frame.frame_id}",
                f"Size: {processed.shape}",
                f"Dropped: {camera.dropped_frames}",
                f"Grayscale: {'ON' if preproc_config.convert_grayscale else 'OFF'}",
                f"CLAHE: {'ON' if preproc_config.apply_clahe else 'OFF'}",
            ]

            # Add video progress
            if camera.is_video:
                progress = camera.video_progress
                if progress:
                    current, total = progress
                    percentage = (current / total * 100) if total > 0 else 0
                    info_text.append(f"Progress: {current}/{total} ({percentage:.1f}%)")

            # Draw info
            y_offset = 30
            for text in info_text:
                cv2.putText(
                    display_img,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                y_offset += 25

            # Show frame
            cv2.imshow("Capture Test", display_img)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord('g'):
                preproc_config.convert_grayscale = not preproc_config.convert_grayscale
                print(f"Grayscale: {'ON' if preproc_config.convert_grayscale else 'OFF'}")

            elif key == ord('c'):
                preproc_config.apply_clahe = not preproc_config.apply_clahe
                if preproc_config.apply_clahe:
                    preprocessor._clahe = cv2.createCLAHE(
                        clipLimit=preproc_config.clahe_clip_limit,
                        tileGridSize=preproc_config.clahe_tile_grid_size
                    )
                print(f"CLAHE: {'ON' if preproc_config.apply_clahe else 'OFF'}")

            elif key == ord('s'):
                filename = f"frame_{frame_count:04d}.png"
                cv2.imwrite(filename, processed.image)
                print(f"Saved: {filename}")

            elif key == ord(' ') and camera.is_video:
                paused = True
                print("Paused (press SPACE to resume)")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Dropped frames: {camera.dropped_frames}")
        print(f"  Final FPS: {camera.fps:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()