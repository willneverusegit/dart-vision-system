"""
Live camera test script.
Tests threaded capture and preprocessing with real-time display.

Usage:
    python scripts/test_camera.py

Controls:
    - 'q': Quit
    - 'g': Toggle grayscale
    - 'c': Toggle CLAHE
    - 'r': Reset ROI
    - 's': Save current frame
"""
import cv2
import sys
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


def main():
    """Run live camera test."""
    print("=" * 60)
    print("Threaded Camera Test")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'g' - Toggle grayscale")
    print("  'c' - Toggle CLAHE")
    print("  's' - Save frame")
    print("=" * 60)

    # Configure camera
    cam_config = CameraConfig(
        index=0,
        width=1280,
        height=720,
        fps=30
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
        logger.error("Failed to start camera")
        return

    print(f"\nCamera properties: {camera.get_camera_properties()}")
    print("Camera started. Press 'q' to quit.\n")

    frame_count = 0

    try:
        while True:
            # Read frame
            frame = camera.read(timeout=1.0)

            if frame is None:
                logger.warning("No frame received")
                continue

            # Preprocess
            processed = preprocessor.process(frame)

            # Display info overlay
            display_img = processed.image.copy()

            # Convert to BGR if grayscale (for colored text overlay)
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            # Add info text
            info_text = [
                f"FPS: {camera.fps:.1f}",
                f"Frame: {frame.frame_id}",
                f"Size: {processed.shape}",
                f"Dropped: {camera.dropped_frames}",
                f"Grayscale: {'ON' if preproc_config.convert_grayscale else 'OFF'}",
                f"CLAHE: {'ON' if preproc_config.apply_clahe else 'OFF'}",
            ]

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
            cv2.imshow("Camera Test", display_img)

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