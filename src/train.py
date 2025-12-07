"""
Training Stage
Trains FraudNet model with class-weighted loss and MLflow tracking.
"""

import numpy as np
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import yaml
import mlflow
import mlflow.pytorch
import os
import sys

from model import get_model


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(params):
    """
    Load processed training and test data.
    
    Args:
        params (dict): Configuration parameters
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as numpy arrays
    """
    print("Loading processed data...")
    
    processed_paths = params['data']['processed']
    
    X_train = np.load(processed_paths['X_train'])
    X_test = np.load(processed_paths['X_test'])
    y_train = np.load(processed_paths['y_train'])
    y_test = np.load(processed_paths['y_test'])
    
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        y_train (np.ndarray): Training labels
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    print("\nCalculating class weights for imbalanced dataset...")
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    print(f"  - Class weights: {class_weights}")
    print(f"  - Positive class weight (fraud): {class_weights[1]:.4f}")
    
    return class_weights_tensor


def accuracy_fn(y_true, y_pred):
    """
    Calculate accuracy.
    
    Args:
        y_true (torch.Tensor): True labels
        y_pred (torch.Tensor): Predicted labels
        
    Returns:
        float: Accuracy percentage
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_model(model, X_train, X_test, y_train, y_test, params, class_weights_tensor):
    """
    Train the FraudNet model with MLflow tracking.
    
    Args:
        model: FraudNet model instance
        X_train, X_test, y_train, y_test: Training and test data
        params (dict): Configuration parameters
        class_weights_tensor (torch.Tensor): Class weights for loss function
        
    Returns:
        nn.Module: Trained model
    """
    # Get training parameters
    epochs = int(os.environ.get('N_EPOCHS_OVERRIDE', params['training']['epochs']))
    learning_rate = params['training']['learning_rate']
    random_seed = params['training']['random_seed']
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Random seed: {random_seed}")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Define loss function with class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])
    
    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # Convert data to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    
    print("\nStarting training...")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        
        # Forward pass
        y_logits = model(X_train_tensor).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        
        # Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train_tensor)
        acc = accuracy_fn(y_true=y_train_tensor, y_pred=y_pred)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluation phase
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test_tensor).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            
            test_loss = loss_fn(test_logits, y_test_tensor)
            test_acc = accuracy_fn(y_true=y_test_tensor, y_pred=test_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        mlflow.log_metric("train_acc", acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss.item(), step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)
        
        # Print progress
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch: {epoch:4d} | Loss: {loss:.5f}, Acc: {acc:.2f}% | "
                  f"Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    
    print("\nTraining completed!")
    return model


def save_model(model, params):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        params (dict): Configuration parameters
    """
    model_path = params['model_path']
    print(f"\nSaving model to {model_path}...")
    
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully!")


def main():
    """Main execution function for training stage."""
    try:
        # Load parameters
        params = load_params()
        
        # Set up MLflow
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        
        # Start MLflow run
        with mlflow.start_run():
            print("=" * 60)
            print("TRAINING STAGE - Credit Card Fraud Detection")
            print("=" * 60)
            
            # Log parameters
            mlflow.log_params({
                "input_features": params['model']['input_features'],
                "hidden_units": params['model']['hidden_units'],
                "output_features": params['model']['output_features'],
                "epochs": int(os.environ.get('N_EPOCHS_OVERRIDE', params['training']['epochs'])),
                "learning_rate": params['training']['learning_rate'],
                "random_seed": params['training']['random_seed'],
                "test_size": params['preprocessing']['test_size']
            })
            
            # Load data
            X_train, X_test, y_train, y_test = load_processed_data(params)
            
            # Calculate class weights
            class_weights_tensor = calculate_class_weights(y_train)
            mlflow.log_param("pos_weight", class_weights_tensor[1].item())
            
            # Create model
            print("\nInitializing FraudNet model...")
            model = get_model(params['model'])
            print(f"Model architecture:\n{model}")
            
            # Train model
            model = train_model(
                model, X_train, X_test, y_train, y_test,
                params, class_weights_tensor
            )
            
            # Save model
            save_model(model, params)
            
            # Log model artifact to MLflow
            mlflow.log_artifact(params['model_path'])
            
            print("\n" + "=" * 60)
            print("✓ Training stage completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error in training stage: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
