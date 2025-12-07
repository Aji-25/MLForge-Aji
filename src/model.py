"""
FraudNet Model Definition
Neural network architecture for credit card fraud detection.
"""

import torch
from torch import nn


class FraudNet(nn.Module):
    """
    4-layer feedforward neural network for binary fraud classification.
    
    Architecture:
    - Input layer -> Hidden layer 1 (ReLU)
    - Hidden layer 1 -> Hidden layer 2 (ReLU)
    - Hidden layer 2 -> Hidden layer 3 (ReLU)
    - Hidden layer 3 -> Output layer (logits)
    """
    
    def __init__(self, input_features, output_features, hidden_units):
        """
        Initialize FraudNet model.
        
        Args:
            input_features (int): Number of input features (30 for credit card dataset)
            output_features (int): Number of output features (1 for binary classification)
            hidden_units (int): Number of units in each hidden layer
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_features)
        """
        return self.linear_layer_stack(x)


def get_model(params):
    """
    Factory function to create FraudNet model from parameters.
    
    Args:
        params (dict): Dictionary containing model configuration with keys:
            - input_features
            - output_features
            - hidden_units
            
    Returns:
        FraudNet: Initialized model instance
    """
    return FraudNet(
        input_features=params['input_features'],
        output_features=params['output_features'],
        hidden_units=params['hidden_units']
    )
