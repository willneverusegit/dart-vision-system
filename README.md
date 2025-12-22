# Dart Vision System

Automatic dart detection and scoring system using Computer Vision.

## Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Architecture

- `src/core/`: Shared data types and utilities
- `src/capture/`: Threaded camera capture and ROI processing
- `src/calibration/`: Perspective correction and scaling
- `src/board/`: Board localization and geometry mapping
- `src/game/`: Game logic and scoring
- `src/detection/`: Motion detection and hit recognition

## Current Status

Phase 1: Initial project setup