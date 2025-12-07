"""
Test Data Pipeline
Tests data loading and preprocessing stages.
"""

import os
import pytest
import pandas as pd
import numpy as np
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_data_load_stage():
    """Test that data loading stage produces expected output."""
    params = load_params()
    interim_path = params['data']['interim']
    
    # Check that interim file exists
    assert os.path.exists(interim_path), f"Interim data file not found: {interim_path}"
    
    # Load and validate interim data
    df = pd.read_csv(interim_path)
    
    # Check shape (should have 31 columns: 30 features + 1 target)
    assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"
    
    # Check that Class column exists
    assert 'Class' in df.columns, "Class column not found in interim data"
    
    # Check no null values
    assert df.isnull().sum().sum() == 0, "Null values found in interim data"
    
    print(f"✓ Data load stage test passed - Shape: {df.shape}")


def test_preprocess_stage():
    """Test that preprocessing stage produces expected outputs."""
    params = load_params()
    processed_paths = params['data']['processed']
    
    # Check that all processed files exist
    assert os.path.exists(processed_paths['X_train']), "X_train.npy not found"
    assert os.path.exists(processed_paths['X_test']), "X_test.npy not found"
    assert os.path.exists(processed_paths['y_train']), "y_train.npy not found"
    assert os.path.exists(processed_paths['y_test']), "y_test.npy not found"
    
    # Load processed data
    X_train = np.load(processed_paths['X_train'])
    X_test = np.load(processed_paths['X_test'])
    y_train = np.load(processed_paths['y_train'])
    y_test = np.load(processed_paths['y_test'])
    
    # Check shapes
    assert X_train.shape[1] == 30, f"Expected 30 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 30, f"Expected 30 features, got {X_test.shape[1]}"
    
    # Check that train/test split ratio is approximately correct
    test_size = params['preprocessing']['test_size']
    total_samples = X_train.shape[0] + X_test.shape[0]
    actual_test_ratio = X_test.shape[0] / total_samples
    
    # Allow 1% tolerance
    assert abs(actual_test_ratio - test_size) < 0.01, \
        f"Test ratio {actual_test_ratio:.3f} differs from expected {test_size}"
    
    # Check that labels are binary (0 or 1)
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train contains non-binary values"
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test contains non-binary values"
    
    print(f"✓ Preprocess stage test passed")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - Test ratio: {actual_test_ratio:.3f}")


def test_data_pipeline_end_to_end():
    """Test complete data pipeline from raw to processed."""
    params = load_params()
    
    # Load raw data
    raw_path = params['data']['raw']
    assert os.path.exists(raw_path), f"Raw data file not found: {raw_path}"
    
    df_raw = pd.read_csv(raw_path)
    
    # Load processed data
    X_train = np.load(params['data']['processed']['X_train'])
    X_test = np.load(params['data']['processed']['X_test'])
    y_train = np.load(params['data']['processed']['y_train'])
    y_test = np.load(params['data']['processed']['y_test'])
    
    # Check that total samples match (raw should equal train + test)
    total_processed = X_train.shape[0] + X_test.shape[0]
    assert total_processed == df_raw.shape[0], \
        f"Sample count mismatch: raw={df_raw.shape[0]}, processed={total_processed}"
    
    # Check class distribution is preserved (stratified split)
    fraud_ratio_raw = df_raw['Class'].sum() / len(df_raw)
    fraud_ratio_train = y_train.sum() / len(y_train)
    fraud_ratio_test = y_test.sum() / len(y_test)
    
    # Allow 5% tolerance for stratification
    assert abs(fraud_ratio_train - fraud_ratio_raw) < 0.05, \
        "Train set class distribution differs significantly from raw data"
    assert abs(fraud_ratio_test - fraud_ratio_raw) < 0.05, \
        "Test set class distribution differs significantly from raw data"
    
    print(f"✓ End-to-end data pipeline test passed")
    print(f"  - Raw samples: {df_raw.shape[0]}")
    print(f"  - Processed samples: {total_processed}")
    print(f"  - Fraud ratio (raw): {fraud_ratio_raw:.4f}")
    print(f"  - Fraud ratio (train): {fraud_ratio_train:.4f}")
    print(f"  - Fraud ratio (test): {fraud_ratio_test:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
