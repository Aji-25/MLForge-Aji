#!/bin/bash
set -e

echo "=== MVDVC Demo: Pipeline ==="

export PATH="/opt/miniconda3/bin:$PATH"

# Clean pipeline state
rm -f .mvdvc/pipeline_state.json
# Ensure data is checked out (dependency for load)
python3 mvdvc.py checkout

echo "[1] First Run (Should run all)"
python3 mvdvc.py repro

echo "[2] Second Run (Should skip all)"
python3 mvdvc.py repro

echo "[3] Modify preprocess (Should rerun preprocess and downstream)"
# Modify src/preprocess.py safely (append a comment)
echo "# touched" >> src/preprocess.py

python3 mvdvc.py repro

# Cleanup modification
# Not strictly necessary but good practice
# git checkout src/preprocess.py # Avoid this as it might not be git tracked in the same way or user might have changes.
