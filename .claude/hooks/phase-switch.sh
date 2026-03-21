#!/bin/bash
# Phase Switch Hook — triggered after git checkout to a new phase branch
# Detects branch change and loads phase context
set -e

PROJECT_DIR="C:/Users/domes/OneDrive/Desktop/FREE_code"
cd "$PROJECT_DIR"

# Get the new branch name from git
BRANCH=$(git branch --show-current 2>/dev/null || echo "")

# Only trigger on phase branches
if [[ "$BRANCH" != feature/phase* ]]; then
  exit 0
fi

# Extract phase number
PHASE_NUM=$(echo "$BRANCH" | grep -oP 'phase\K[0-9]+' || echo "")
if [ -z "$PHASE_NUM" ]; then
  exit 0
fi

echo "=== PHASE SWITCH DETECTED ==="
echo "Neuer Branch: $BRANCH"

# Map phase to task file and skills
case "$PHASE_NUM" in
  1) SKILLS="/dart-api, /dart-frontend, /dart-scoring"; AGENTS="backend-coder, frontend-coder" ;;
  2) SKILLS="/dart-calibration"; AGENTS="backend-coder, frontend-coder" ;;
  3) SKILLS="/dart-api"; AGENTS="backend-coder, frontend-coder" ;;
  4) SKILLS="/dart-vision-pipeline, /dart-scoring"; AGENTS="backend-coder" ;;
  5) SKILLS="/dart-frontend, /dart-calibration"; AGENTS="backend-coder, frontend-coder" ;;
  6) SKILLS="/dart-scoring"; AGENTS="backend-coder" ;;
  7) SKILLS="/dart-calibration, /dart-vision-pipeline"; AGENTS="backend-coder, frontend-coder" ;;
  8) SKILLS="/dart-vision-pipeline"; AGENTS="backend-coder" ;;
  9) SKILLS="/dart-scoring"; AGENTS="backend-coder, frontend-coder" ;;
  10) SKILLS="/dart-calibration, /dart-vision-pipeline"; AGENTS="backend-coder" ;;
  *) SKILLS=""; AGENTS="" ;;
esac

echo "Relevante Skills: $SKILLS"
echo "Relevante Agents: $AGENTS"

# Show phase task file
PHASE_FILE=""
for f in tasks/phase${PHASE_NUM}*.md; do
  if [ -f "$f" ]; then
    PHASE_FILE="$f"
    break
  fi
done

if [ -n "$PHASE_FILE" ]; then
  echo ""
  echo "--- PHASE TASKS ($PHASE_FILE) ---"
  # Show task titles and status
  grep -E "^###|Status" "$PHASE_FILE" 2>/dev/null | head -20
fi

# Update agentic-os session-summary if it exists
MEMORY_DIR=".agent-memory"
if [ -d "$MEMORY_DIR" ]; then
  echo ""
  echo "Aktualisiere agentic-os Session-Summary..."
  cat > "$MEMORY_DIR/session-summary.md" << SUMMARY
# Session Summary — Dart Vision System

## Aktiver Branch
$BRANCH

## Aktive Phase
Phase $PHASE_NUM

## Relevante Skills
$SKILLS

## Relevante Agents
$AGENTS

## Nächste Schritte
Siehe $PHASE_FILE für offene Tasks.
SUMMARY
  echo "Session-Summary aktualisiert."
fi

echo "=== PHASE KONTEXT GELADEN ==="
