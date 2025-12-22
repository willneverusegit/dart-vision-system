# scripts/generate_charuco.py
"""
Generate ChArUco calibration board for printing.

Usage:
    python scripts/generate_charuco.py
    python scripts/generate_charuco.py --output charuco_board.png
    python scripts/generate_charuco.py --size A4
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import CharucoCalibrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ChArUco calibration board for printing"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="charuco_board.png",
        help="Output image file (default: charuco_board.png)"
    )

    parser.add_argument(
        "--size",
        choices=["A4", "Letter", "custom"],
        default="A4",
        help="Paper size preset (default: A4)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for printing (default: 300)"
    )

    return parser.parse_args()


def main():
    """Generate ChArUco board."""
    args = parse_args()

    # Calculate board size based on paper and DPI
    if args.size == "A4":
        # A4: 210 x 297 mm
        width_mm, height_mm = 210, 297
    elif args.size == "Letter":
        # Letter: 8.5 x 11 inches = 215.9 x 279.4 mm
        width_mm, height_mm = 215.9, 279.4
    else:
        # Custom: use default
        width_mm, height_mm = 210, 297

    # Convert to pixels
    width_px = int(width_mm / 25.4 * args.dpi)
    height_px = int(height_mm / 25.4 * args.dpi)

    # Board configuration (5x7 squares)
    squares_x, squares_y = 5, 7

    # Calculate square size to fit paper with margins
    margin_px = int(20 / 25.4 * args.dpi)  # 20mm margins
    available_width = width_px - 2 * margin_px
    available_height = height_px - 2 * margin_px

    square_length = min(
        available_width // squares_x,
        available_height // squares_y
    )
    marker_length = int(square_length * 0.75)

    print(f"Generating ChArUco board:")
    print(f"  Paper: {args.size} ({width_mm:.1f} x {height_mm:.1f} mm)")
    print(f"  Resolution: {width_px} x {height_px} px @ {args.dpi} DPI")
    print(f"  Squares: {squares_x} x {squares_y}")
    print(f"  Square size: {square_length} px")
    print(f"  Marker size: {marker_length} px")

    # Generate board
    board_image = CharucoCalibrator.generate_board_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        output_path=args.output
    )

    print(f"\n✓ ChArUco board saved to: {args.output}")
    print(f"\nPrinting instructions:")
    print(f"  1. Print at {args.dpi} DPI (do NOT scale)")
    print(f"  2. Use thick paper or mount on cardboard")
    print(f"  3. Ensure flat surface for best results")


if __name__ == "__main__":
    main()