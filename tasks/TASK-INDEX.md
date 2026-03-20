# Dart Vision System — Task Index

## Priorisierung (nach Impact/Aufwand Score)

| # | Phase | Score | Status |
|---|-------|-------|--------|
| 1 | [MVP: Basic UI + Manual Scoring](phase1-mvp.md) | 2.50 | TODO |
| 2 | [Persistenz & Profilmanagement](phase3-persistence.md) | 2.00 | TODO |
| 3 | [Single-Cam Kalibrierung & ROI](phase2-calibration.md) | 1.67 | TODO |
| 4 | [Top-Down View & Score-Overlay](phase5-topdown-view.md) | 1.67 | TODO |
| 5 | [Multiplayer & Spielmodi](phase6-multiplayer.md) | 1.50 | TODO |
| 6 | [Single-Cam Treffererkennung](phase4-hit-detection.md) | 1.00 | TODO |
| 7 | [Confidence-Modul](phase9-confidence.md) | 1.00 | TODO |
| 8 | [Stereo-Kalibrierung & Triangulation](phase7-stereo.md) | 0.80 | TODO |
| 9 | [Eco/Debug-Modi & Performance](phase8-performance.md) | 0.75 | TODO |
| 10 | [Dritte Kamera & Erweiterungen](phase10-extensions.md) | 0.40 | TODO |

## Abhängigkeiten

```
Phase 1 (MVP) ──────┬──> Phase 2 (Persistenz)
                     ├──> Phase 3 (Kalibrierung) ──> Phase 4 (Top-Down)
                     └──> Phase 5 (Multiplayer)
Phase 3 (Kalibrierung) ──> Phase 6 (Treffererkennung) ──> Phase 7 (Confidence)
Phase 3 (Kalibrierung) ──> Phase 8 (Stereo)
Phase 6 + Phase 8 ──> Phase 9 (Performance)
Phase 8 (Stereo) ──> Phase 10 (Erweiterungen)
```

## Legende
- **TODO** — noch nicht begonnen
- **IN PROGRESS** — in Arbeit
- **DONE** — abgeschlossen
