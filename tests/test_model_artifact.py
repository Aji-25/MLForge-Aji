"""
Test Model Artifacts
Tests evaluation stage outputs and model artifacts.
"""

import os
import pytest
import json
import yaml
from PIL import Image


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_metrics_file_exists():
    """Test that metrics.json file is created."""
    params = load_params()
    metrics_path = params['evaluation']['metrics_path']
    
    assert os.path.exists(metrics_path), f"Metrics file not found: {metrics_path}"
    
    print(f"✓ Metrics file exists: {metrics_path}")


def test_metrics_content():
    """Test that metrics.json contains required keys and valid values."""
    params = load_params()
    metrics_path = params['evaluation']['metrics_path']
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Check required keys
    required_keys = ['precision', 'recall', 'f1_score', 'accuracy']
    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"
    
    # Check that values are valid (between 0 and 1)
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), f"{key} is not a number"
        assert 0 <= value <= 1, f"{key} value {value} is out of range [0, 1]"
    
    print(f"✓ Metrics content validated")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-score: {metrics['f1_score']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")


def test_confusion_matrix_exists():
    """Test that confusion matrix plot is created."""
    params = load_params()
    cm_path = params['evaluation']['confusion_matrix_path']
    
    assert os.path.exists(cm_path), f"Confusion matrix plot not found: {cm_path}"
    
    print(f"✓ Confusion matrix plot exists: {cm_path}")


def test_confusion_matrix_is_valid_image():
    """Test that confusion matrix is a valid image file."""
    params = load_params()
    cm_path = params['evaluation']['confusion_matrix_path']
    
    # Try to open as image
    try:
        img = Image.open(cm_path)
        width, height = img.size
        
        # Check reasonable dimensions
        assert width > 0 and height > 0, "Image has invalid dimensions"
        assert width >= 400 and height >= 300, "Image is too small"
        
        print(f"✓ Confusion matrix is valid image")
        print(f"  - Dimensions: {width}x{height}")
        print(f"  - Format: {img.format}")
        
    except Exception as e:
        pytest.fail(f"Failed to open confusion matrix as image: {str(e)}")


def test_model_file_exists():
    """Test that trained model file exists."""
    params = load_params()
    model_path = params['model_path']
    
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    # Check file size (should be > 0)
    file_size = os.path.getsize(model_path)
    assert file_size > 0, "Model file is empty"
    
    print(f"✓ Model file exists: {model_path}")
    print(f"  - File size: {file_size / 1024:.2f} KB")


def test_all_evaluation_artifacts():
    """Test that all evaluation artifacts are present."""
    params = load_params()
    
    artifacts = {
        'Model': params['model_path'],
        'Metrics': params['evaluation']['metrics_path'],
        'Confusion Matrix': params['evaluation']['confusion_matrix_path']
    }
    
    missing = []
    for name, path in artifacts.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    
    assert len(missing) == 0, f"Missing artifacts: {', '.join(missing)}"
    
    print(f"✓ All evaluation artifacts present:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
