# Dart Vision System

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

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