# Vereinfachte Dart-Erkennung

## Problem
Je mehr Validierungen ich eingebaut habe, desto weniger funktioniert die Hit-Detection.

## Lösung: SimpleHitDetector

Ein drastisch vereinfachter Ansatz, der sich auf das Wesentliche konzentriert:

### Alter Ansatz (EnhancedHitDetector) - ZU KOMPLEX ❌

```
Motion Detection
    ↓
State Machine (IDLE → WATCHING → CONFIRMING → COOLDOWN)
    ↓
Quiet Frames Requirement (confirming_quiet_frames)
    ↓
Temporal Confirmation (3+ frames)
    ↓
Reverse Motion Check
    ↓
Velocity Settling Check
    ↓
Entry Angle Validation
    ↓
Spatial Spacing Check (min_pixel_drift)
    ↓
Ring-based Validation
    ↓
Cooldown Zone Check
    ↓
Frame Spacing Check (min_frames_since_hit)
    ↓
Hit! (vielleicht... wenn alles passt)
```

**Probleme:**
- Zu viele Validierungen blockieren echte Hits
- Komplexe State Machine ist fehleranfällig
- Über-Engineering führt zu mehr Bugs
- Schwer zu debuggen

### Neuer Ansatz (SimpleHitDetector) - EINFACH ✅

```
Motion Detection
    ↓
Motion gestoppt?
    ↓
Konturen finden
    ↓
Dart-Spitze bestimmen
    ↓
Score berechnen
    ↓
Hit! ✓
```

**Vorteile:**
- ✅ Weniger Code = weniger Bugs
- ✅ Einfach zu verstehen und debuggen
- ✅ Fokus auf was wirklich funktioniert
- ✅ Keine Over-Engineering

## Verwendung

### Einfach (empfohlen):

```python
from src.board import DartboardMapper
from src.detection import SimpleHitDetector

mapper = DartboardMapper(calibration_file="calibration.json")
detector = SimpleHitDetector(mapper)

# Das wars!
hit = detector.detect(frame)
if hit:
    print(f"Treffer! {hit.score} Punkte")
```

### Mit Custom Motion Config:

```python
from src.detection import SimpleHitDetector
from src.detection.motion import MotionConfig

motion_config = MotionConfig(
    threshold=20,    # Motion sensitivity
    min_area=50,     # Minimum motion blob size
)

detector = SimpleHitDetector(mapper, motion_config)
```

## Was wurde entfernt?

❌ **Gelöscht (unnötig):**
- Komplexe State Machine (4 States → 2 einfache Flags)
- `confirmation_frames` (3+ Frame Bestätigung)
- `confirming_quiet_frames` (Warten auf Ruhe)
- Reverse motion check
- Velocity settling validation
- Entry angle plausibility check
- Ring-based validation
- Komplexe spatial/temporal spacing
- Candidate tracking über viele Frames

✅ **Behalten (wichtig):**
- Motion Detection (wissen wann geworfen wird)
- Background Subtraction (Dart von Background unterscheiden)
- Kontur-Erkennung mit Größenfilter
- Dart-Spitzen-Erkennung
- Einfacher Cooldown (1 Sekunde)
- Mindestabstand zum letzten Hit (20 Pixel)

## Kernphilosophie

> **"Simple is better than complex."** - Zen of Python

Statt 10+ Validierungen die alle erfüllt sein müssen:
- Nutze Motion Detection um zu wissen WANN zu suchen ist
- Finde Konturen die wie ein Dart aussehen (Größe)
- Bestimme die Spitze
- Fertig!

## Performance

Der neue Detector ist:
- **Schneller** (weniger Checks)
- **Robuster** (weniger kann schiefgehen)
- **Einfacher** (leichter zu debuggen)

## Wechsel zum Simple Detector

Ändere einfach die Import-Zeile:

```python
# Alt:
from src.detection import HitDetector

# Neu:
from src.detection import SimpleHitDetector as HitDetector
```

Fertig! Keine Config-Änderungen nötig.
