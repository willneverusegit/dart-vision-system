# Phase 3 — Persistenz & Profilmanagement

**Impact**: 4 | **Aufwand**: 2 | **Score**: 2.00
**Abhängigkeit**: Phase 1 abgeschlossen
**Skill**: `/dart-api`
**Agent**: `backend-coder` (T1, T3), `frontend-coder` (T2)

## Tasks

### P3-T1: Profil-CRUD API
- **Status**: TODO
- **Beschreibung**: REST-Endpoints für Kamera-Profile (JSON-basiert)
- **Dateien**:
  - `backend/api/profiles.py` — GET/POST/DELETE /api/profiles
  - `backend/services/profile_store.py` — Laden/Speichern in `backend/data/profiles/`
  - `backend/tests/test_profiles.py`
- **Akzeptanzkriterien**:
  - Profile erstellen, listen, laden, löschen
  - JSON-Dateien in `data/profiles/` korrekt geschrieben
  - Validierung: ungültige Profile werden abgelehnt

### P3-T2: Profil-Management UI
- **Status**: TODO
- **Beschreibung**: Profil-Auswahl in der Setup-Seite
- **Dateien**:
  - `frontend/js/setup-ui.js` — Profil-Dropdown, Laden/Speichern Buttons
- **Akzeptanzkriterien**:
  - Dropdown zeigt alle gespeicherten Profile
  - Profil laden setzt Kamera-Einstellungen
  - Neues Profil speichern mit Namenseingabe

### P3-T3: Auto-Load beim Start
- **Status**: TODO
- **Beschreibung**: Letztes aktives Profil automatisch laden
- **Dateien**:
  - `backend/services/profile_store.py` — get_last_profile(), set_active()
- **Akzeptanzkriterien**:
  - Beim Server-Start wird letztes Profil geladen
  - Kein Fehler wenn kein Profil existiert
