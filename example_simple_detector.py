"""
Example: How to use the simplified hit detector.

This shows the difference between the old complex approach
and the new simple one.
"""
from src.board import DartboardMapper
from src.detection import SimpleHitDetector
from src.detection.motion import MotionConfig

# Setup (same as before)
mapper = DartboardMapper(calibration_file="path/to/calibration.json")

# OLD WAY (too complex):
# from src.detection import HitDetector, HitDetectionConfig, StateConfig
# config = HitDetectionConfig(
#     confirmation_frames=3,
#     state_config=StateConfig(
#         confirming_quiet_frames=2,
#         watching_duration_sec=1.0,
#         confirming_duration_sec=1.5,
#         cooldown_duration_sec=1.0,
#     ),
#     min_frames_since_hit=2,
#     min_pixel_drift_since_hit=8.0,
#     entry_angle_tolerance_deg=70.0,
#     # ... and many more settings!
# )
# detector = HitDetector(mapper, config)

# NEW WAY (simple!):
detector = SimpleHitDetector(mapper)

# That's it! No complex config needed.
# You can optionally customize motion detection:
motion_config = MotionConfig(
    threshold=20,  # Motion sensitivity
    min_area=50,   # Minimum motion blob size
)
detector = SimpleHitDetector(mapper, motion_config)

# Usage is the same:
# hit = detector.detect(frame)
# if hit:
#     print(f"Hit! Score: {hit.score}")

print("SimpleHitDetector ready to use!")
print("\nPhilosophy:")
print("  ✓ Motion detected → Start watching")
print("  ✓ Motion stopped → Find contours")
print("  ✓ Find dart tip → Score it")
print("  ✓ Done!")
print("\nNo over-engineering, just what works.")
