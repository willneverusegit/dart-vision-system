# Phase 9 — Confidence-Modul

**Impact**: 3 | **Aufwand**: 3 | **Score**: 1.00
**Abhängigkeit**: Phase 4 (Treffererkennung)
**Skill**: `/dart-scoring`
**Agent**: `backend-coder` (T1), `frontend-coder` (T2)

## Tasks

### P9-T1: Confidence-Berechnung
- **Status**: TODO
- **Beschreibung**: Vertrauensmaß aus Draht-Distanz, Detektionsqualität, Triangulationsfehler
- **Dateien**:
  - `backend/scoring/confidence.py` — calculate_confidence()
  - `backend/tests/test_confidence.py`
- **Akzeptanzkriterien**:
  - Confidence 0.0-1.0 berechnet
  - Faktoren: Distanz zum nächsten Draht (40%), Detektionsqualität (40%), Triangulation (20%)
  - < 0.6 löst manuelle Review aus

### P9-T2: Confidence UI
- **Status**: TODO
- **Beschreibung**: Visueller Indikator + manueller Review-Dialog
- **Dateien**:
  - `frontend/js/game-ui.js` — Erweiterung: showConfidence(), showReviewDialog()
- **Akzeptanzkriterien**:
  - Farbiger Ring um Treffer-Marker (grün > 0.8, gelb > 0.6, rot < 0.6)
  - Dialog bei niedriger Confidence: "Treffer bestätigen? [T20] [T5] [Miss]"
  - Manuell korrigierter Score wird übernommen
