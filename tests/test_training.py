"""
Test Training Stage
Tests model training with epoch override for fast CI/CD validation.
"""

import os
import pytest
import torch
import yaml
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import get_model


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_training_with_epoch_override():
    """Test training stage with N_EPOCHS_OVERRIDE=1 for fast validation."""
    params = load_params()
    
    # Set environment variable to override epochs
    os.environ['N_EPOCHS_OVERRIDE'] = '1'
    
    # Run training script
    import subprocess
    result = subprocess.run(
        ['python', 'src/train.py'],
        capture_output=True,
        text=True
    )
    
    # Check that training completed successfully
    assert result.returncode == 0, f"Training failed with error:\n{result.stderr}"
    
    # Check that model file was created
    model_path = params['model_path']
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    print(f"✓ Training test passed - Model saved to {model_path}")
    
    # Clean up environment variable
    del os.environ['N_EPOCHS_OVERRIDE']


def test_model_can_be_loaded():
    """Test that saved model can be loaded successfully."""
    params = load_params()
    model_path = params['model_path']
    
    # Check model file exists
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    # Create model instance
    model = get_model(params['model'])
    
    # Load state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.eval()
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, params['model']['input_features'])
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (1, params['model']['output_features']), \
        f"Unexpected output shape: {output.shape}"
    
    print(f"✓ Model loading test passed")
    print(f"  - Model architecture validated")
    print(f"  - Forward pass successful")


def test_model_architecture():
    """Test that model has correct architecture."""
    params = load_params()
    
    # Create model
    model = get_model(params['model'])
    
    # Check model type
    from model import FraudNet
    assert isinstance(model, FraudNet), "Model is not an instance of FraudNet"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params == total_params, "Some parameters are not trainable"
    
    print(f"✓ Model architecture test passed")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
