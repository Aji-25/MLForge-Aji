#!/bin/bash
set -e

echo "=== MVDVC Demo: Data Versioning ==="

# Clean previous run
rm -rf .mvdvc
rm -rf /tmp/mvdvc_remote
git checkout data/raw/creditcard.csv 2>/dev/null || true

echo "[1] Init"
python3 mvdvc.py init

echo "[2] Add data"
python3 mvdvc.py add data/raw/creditcard.csv

echo "[3] Remote add"
mkdir -p /tmp/mvdvc_remote
python3 mvdvc.py remote add origin local /tmp/mvdvc_remote

echo "[4] Push"
python3 mvdvc.py push

echo "[5] Delete local data"
rm data/raw/creditcard.csv
if [ ! -f data/raw/creditcard.csv ]; then
    echo "Files deleted successfully."
fi

echo "[6] Checkout (Restore)"
python3 mvdvc.py checkout

if [ -f data/raw/creditcard.csv ]; then
    echo "SUCCESS: Data restored."
else
    echo "FAILURE: Data not restored."
    exit 1
fi
