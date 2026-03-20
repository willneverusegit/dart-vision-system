# Phase 6 — Multiplayer & Spielmodi

**Impact**: 3 | **Aufwand**: 2 | **Score**: 1.50
**Abhängigkeit**: Phase 1 (MVP)
**Skill**: `/dart-scoring`
**Agent**: `backend-coder` (T1-T2), `frontend-coder` (T3 UI-Anpassung)

## Tasks

### P6-T1: Player Management
- **Status**: TODO
- **Beschreibung**: Spieler hinzufügen/entfernen, Zugreihenfolge
- **Dateien**:
  - `backend/scoring/game_engine.py` — add_player(), next_turn(), reset_turn()
  - `backend/tests/test_game_engine.py`
- **Akzeptanzkriterien**:
  - 1-8 Spieler unterstützt
  - Automatischer Wechsel nach 3 Würfen
  - Spieler-Reihenfolge korrekt

### P6-T2: 301/501 Rules Engine
- **Status**: TODO
- **Beschreibung**: Double-Out, Bust-Logik, Checkout-Erkennung
- **Dateien**:
  - `backend/scoring/game_engine.py` — process_throw_x01(), check_bust()
  - `backend/tests/test_game_engine.py`
- **Akzeptanzkriterien**:
  - Bust bei Überwerfen oder Rest=1
  - Double-Out Pflicht für Checkout
  - Winner-Detection bei Rest=0 mit Double

### P6-T3: Cricket Mode
- **Status**: TODO
- **Beschreibung**: Grundlegende Cricket-Scoring-Regeln
- **Dateien**:
  - `backend/scoring/game_engine.py` — process_throw_cricket()
- **Akzeptanzkriterien**:
  - Felder 15-20 + Bull trackbar (offen/geschlossen)
  - Punkte nur wenn Gegner-Feld noch offen
  - Sieg wenn alle Felder geschlossen + höchster Score
