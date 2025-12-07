#!/bin/bash
# Quick setup script for MLForge Credit Card Fraud Detection

echo "=========================================="
echo "MLForge Credit Card Fraud Detection Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify data file
echo ""
echo "Verifying data file..."
if [ -f "data/raw/creditcard.csv" ]; then
    echo "✓ Data file found: data/raw/creditcard.csv"
    ls -lh data/raw/creditcard.csv
else
    echo "✗ Warning: data/raw/creditcard.csv not found"
    echo "  Would you like to generate dummy data for testing? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
        python src/make_dummy_data.py
    else
        echo "  Please place the creditcard.csv file in data/raw/"
    fi
fi

# Initialize DVC (optional)
echo ""
echo "Initializing DVC..."
dvc init --no-scm || echo "DVC already initialized or not needed"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the pipeline: dvc repro"
echo "2. View MLflow UI: mlflow ui"
echo "3. Run tests: pytest tests/ -v"
echo ""
