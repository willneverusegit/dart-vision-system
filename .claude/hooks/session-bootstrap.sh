#!/bin/bash
# Session Bootstrap Hook for Dart Vision System
# Reads current branch, determines active phase, outputs context briefing
set -e

PROJECT_DIR="C:/Users/domes/OneDrive/Desktop/FREE_code"
cd "$PROJECT_DIR"

echo "=== DART VISION SESSION BOOTSTRAP ==="

# 1. Current branch → active phase
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "Branch: $BRANCH"

# 2. Determine phase from branch name
PHASE=""
PHASE_FILE=""
case "$BRANCH" in
  feature/phase1*) PHASE="Phase 1 — MVP"; PHASE_FILE="tasks/phase1-mvp.md" ;;
  feature/phase2*) PHASE="Phase 2 — Calibration"; PHASE_FILE="tasks/phase2-calibration.md" ;;
  feature/phase3*) PHASE="Phase 3 — Persistence"; PHASE_FILE="tasks/phase3-persistence.md" ;;
  feature/phase4*) PHASE="Phase 4 — Hit Detection"; PHASE_FILE="tasks/phase4-hit-detection.md" ;;
  feature/phase5*) PHASE="Phase 5 — Top-Down View"; PHASE_FILE="tasks/phase5-topdown-view.md" ;;
  feature/phase6*) PHASE="Phase 6 — Multiplayer"; PHASE_FILE="tasks/phase6-multiplayer.md" ;;
  feature/phase7*) PHASE="Phase 7 — Stereo"; PHASE_FILE="tasks/phase7-stereo.md" ;;
  feature/phase8*) PHASE="Phase 8 — Performance"; PHASE_FILE="tasks/phase8-performance.md" ;;
  feature/phase9*) PHASE="Phase 9 — Confidence"; PHASE_FILE="tasks/phase9-confidence.md" ;;
  feature/phase10*) PHASE="Phase 10 — Extensions"; PHASE_FILE="tasks/phase10-extensions.md" ;;
  master|main) PHASE="Kein Phase-Branch aktiv"; PHASE_FILE="" ;;
  *) PHASE="Unbekannter Branch"; PHASE_FILE="" ;;
esac

echo "Aktive Phase: $PHASE"

# 3. Show TASK-INDEX summary (TODO/IN PROGRESS items only)
if [ -f "tasks/TASK-INDEX.md" ]; then
  echo ""
  echo "--- TASK STATUS ---"
  grep -E "(TODO|IN PROGRESS)" tasks/TASK-INDEX.md 2>/dev/null || echo "Alle Tasks erledigt oder kein Status gefunden."
fi

# 4. Show current phase tasks if on a phase branch
if [ -n "$PHASE_FILE" ] && [ -f "$PHASE_FILE" ]; then
  echo ""
  echo "--- AKTIVE PHASE TASKS ---"
  grep -E "^\*\*Status\*\*:" "$PHASE_FILE" 2>/dev/null | head -10 || true
fi

# 5. Last commit info
echo ""
echo "--- LETZTER COMMIT ---"
git log --oneline -3 2>/dev/null || echo "Keine Commits."

# 6. Test status
echo ""
echo "--- TEST STATUS ---"
if [ -d "backend/tests" ] && command -v python &> /dev/null; then
  python -m pytest backend/tests/ --tb=no -q 2>&1 | tail -1 || echo "Tests nicht ausführbar."
else
  echo "Keine Tests vorhanden oder Python nicht verfügbar."
fi

echo ""
echo "=== BOOTSTRAP COMPLETE ==="
