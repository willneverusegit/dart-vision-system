---
name: dart-scoring
description: Guidance for scoring engine (board geometry, polar field mapping, game modes 301/501, confidence)
user_invocable: true
---

# Dart Scoring Skill

## Wann verwenden
Bei Arbeit an Board-Geometrie, Feld-Zuordnung, Punkteberechnung, Spielmodi, Confidence.

## Board-Geometrie (Standard-Dartboard)

```python
# Sektoren im Uhrzeigersinn ab 12-Uhr-Position
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SECTOR_ANGLE = 360 / 20  # 18°

# Ring-Radien (mm vom Mittelpunkt)
BULL_RADIUS = 6.35        # Inner Bull (50 Punkte)
OUTER_BULL_RADIUS = 15.9  # Outer Bull (25 Punkte)
TRIPLE_INNER = 99.0       # Triple-Ring Innenkante
TRIPLE_OUTER = 107.0      # Triple-Ring Außenkante
DOUBLE_INNER = 162.0      # Double-Ring Innenkante
DOUBLE_OUTER = 170.0      # Double-Ring Außenkante (Board-Rand)
```

## Feld-Zuordnung (Polar-Koordinaten)
```python
import math

def get_score(x: float, y: float) -> tuple[int, str, int]:
    """Berechnet Score aus Board-Koordinaten (0,0 = Mittelpunkt).

    Returns: (punkte, ring_name, sektor_nummer)
    """
    dist = math.sqrt(x**2 + y**2)
    angle = math.degrees(math.atan2(x, y)) % 360  # 0° = oben

    # Ring bestimmen
    if dist <= BULL_RADIUS:
        return (50, "double_bull", 0)
    elif dist <= OUTER_BULL_RADIUS:
        return (25, "single_bull", 0)
    elif dist > DOUBLE_OUTER:
        return (0, "miss", 0)

    # Sektor bestimmen (mit Offset für halben Sektor)
    sector_idx = int((angle + SECTOR_ANGLE / 2) % 360 / SECTOR_ANGLE)
    sector_value = SECTORS[sector_idx]

    if TRIPLE_INNER <= dist <= TRIPLE_OUTER:
        return (sector_value * 3, "triple", sector_value)
    elif DOUBLE_INNER <= dist <= DOUBLE_OUTER:
        return (sector_value * 2, "double", sector_value)
    else:
        return (sector_value, "single", sector_value)
```

## Spielmodi

### 301 / 501
```python
def process_throw_x01(game: GameState, hit: HitEvent) -> GameState:
    player = game.players[game.current_player]
    remaining = player.remaining_points - hit.score

    if remaining < 0 or (remaining == 0 and hit.ring != "double"):
        # Bust: Runde ungültig, Punkte zurücksetzen
        return reset_turn(game)
    elif remaining == 0 and hit.ring == "double":
        # Checkout!
        player.remaining_points = 0
        return finish_game(game, winner=player)
    else:
        player.remaining_points = remaining
        game.history.append(hit)
        return advance_turn(game)
```

### Freies Spiel
- Einfach Punkte addieren, kein Checkout erforderlich

## Confidence-Berechnung
```python
def calculate_confidence(
    tip_point: tuple[float, float],
    board_model: BoardModel,
    detection_quality: float,  # 0-1 aus Kontur-Analyse
    multi_cam_error: float | None = None  # Triangulationsfehler in mm
) -> float:
    # Distanz zum nächsten Draht/Ring-Grenze
    dist_to_boundary = get_min_boundary_distance(tip_point, board_model)
    boundary_conf = min(dist_to_boundary / 5.0, 1.0)  # 5mm = volle Konfidenz

    confidence = boundary_conf * 0.4 + detection_quality * 0.4
    if multi_cam_error is not None:
        triangulation_conf = max(0, 1 - multi_cam_error / 3.0)  # 3mm = null
        confidence = confidence * 0.5 + triangulation_conf * 0.5

    return round(confidence, 2)
    # < 0.6: manuelle Bestätigung im UI auslösen
```

## Dateien
- `backend/scoring/board_model.py` — Geometrie + Feld-Mapping
- `backend/scoring/game_engine.py` — Spiellogik (301/501/Cricket)
- `backend/scoring/confidence.py` — Confidence-Berechnung
- `backend/tests/test_scoring.py`
