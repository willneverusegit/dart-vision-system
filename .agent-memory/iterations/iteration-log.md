# Iteration Log

## [2026-03-21 Session] Phase 4: Single-Cam Treffererkennung

**Category:** architecture | **Severity:** N/A (feature) | **Attempts:** 1

**Problem:** Phase 4 (Hit Detection) fehlte komplett — kein Scoring-Modul, kein Detection-Loop, WebSocket nur Echo-Stub.

**Root Cause:** Noch nicht implementiert (geplanter Meilenstein).

**Solution:** 6 neue Dateien erstellt:
- `backend/scoring/board_model.py` — Field-Mapping + Score-Berechnung
- `backend/vision/detection.py` — BackgroundModel + Motion Detection
- `backend/vision/pipeline.py` — Contour-Pipeline + DetectionLoop
- `backend/api/detection.py` — REST-Endpoints (background/start/stop/status)
- `backend/api/websocket.py` — ConnectionManager mit broadcast() (Upgrade)
- 4 Testdateien (34 neue Tests)

**Failed Approaches:** Keine — sauberer Durchlauf.

**Takeaway:** Design-Spec vor Implementierung schreiben spart Trial-and-Error. Parallele Modul-Erstellung (Scoring + Vision + API) funktioniert gut wenn Interfaces klar definiert sind.

---
