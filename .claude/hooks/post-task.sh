#!/bin/bash
# Post-task hook: Runs pytest after backend changes
set -e

echo "=== Post-Task: Running Tests ==="

if [ -d "backend/tests" ] && find backend/tests -name "test_*.py" -type f | head -1 | grep -q .; then
  echo "Running pytest..."
  python -m pytest backend/tests/ -v --tb=short || { echo "WARN: Some tests failed."; exit 1; }
else
  echo "No tests found yet. Skipping."
fi

echo "=== Tests complete ==="
