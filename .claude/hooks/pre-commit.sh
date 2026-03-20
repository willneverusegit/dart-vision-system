#!/bin/bash
# Pre-commit hook: Lint und Format-Check für Python-Backend
set -e

echo "=== Dart Vision Pre-Commit Check ==="

# Python lint check
if [ -d "backend" ] && find backend -name "*.py" -type f | head -1 | grep -q .; then
  echo "Running ruff check..."
  ruff check backend/ || { echo "FAIL: ruff check failed. Run 'ruff check --fix backend/' to fix."; exit 1; }

  echo "Running ruff format check..."
  ruff format --check backend/ || { echo "FAIL: Code not formatted. Run 'ruff format backend/' to fix."; exit 1; }
fi

# JS lint check (optional, wenn eslint vorhanden)
if command -v npx &> /dev/null && [ -f "frontend/.eslintrc.json" ]; then
  echo "Running eslint..."
  npx eslint frontend/js/ || { echo "WARN: eslint issues found."; }
fi

echo "=== All checks passed ==="
