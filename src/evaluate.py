"""
Evaluation Stage
Evaluates trained model and generates metrics and visualizations.
"""

import numpy as np
import torch
import yaml
import json
import mlflow
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import get_model


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_test_data(params):
    """
    Load test data.
    
    Args:
        params (dict): Configuration parameters
        
    Returns:
        tuple: (X_test, y_test) as numpy arrays
    """
    print("Loading test data...")
    
    processed_paths = params['data']['processed']
    
    X_test = np.load(processed_paths['X_test'])
    y_test = np.load(processed_paths['y_test'])
    
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    
    return X_test, y_test


def load_trained_model(params):
    """
    Load trained model from disk.
    
    Args:
        params (dict): Configuration parameters
        
    Returns:
        nn.Module: Loaded model
    """
    model_path = params['model_path']
    print(f"\nLoading trained model from {model_path}...")
    
    # Create model instance
    model = get_model(params['model'])
    
    # Load state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Model loaded successfully!")
    return model


def generate_predictions(model, X_test):
    """
    Generate predictions on test data.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        
    Returns:
        np.ndarray: Binary predictions
    """
    print("\nGenerating predictions...")
    
    with torch.inference_mode():
        X_test_tensor = torch.from_numpy(X_test).float()
        test_logits = model(X_test_tensor).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
    
    predictions = test_pred.numpy()
    print(f"Predictions generated: {len(predictions)} samples")
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    print("\nCalculating metrics...")
    
    metrics = {
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred))
    }
    
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-score: {metrics['f1_score']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def save_metrics(metrics, params):
    """
    Save metrics to JSON file.
    
    Args:
        metrics (dict): Evaluation metrics
        params (dict): Configuration parameters
    """
    metrics_path = params['evaluation']['metrics_path']
    print(f"\nSaving metrics to {metrics_path}...")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved successfully!")


def plot_confusion_matrix(y_true, y_pred, params):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        params (dict): Configuration parameters
        
    Returns:
        str: Path to saved confusion matrix plot
    """
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix - Credit Card Fraud Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    cm_path = params['evaluation']['confusion_matrix_path']
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {cm_path}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    return cm_path


def main():
    """Main execution function for evaluation stage."""
    try:
        # Load parameters
        params = load_params()
        
        # Set up MLflow
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        
        # Start nested MLflow run for evaluation
        with mlflow.start_run(run_name="evaluation", nested=True):
            print("=" * 60)
            print("EVALUATION STAGE - Credit Card Fraud Detection")
            print("=" * 60)
            
            # Load test data
            X_test, y_test = load_test_data(params)
            
            # Load trained model
            model = load_trained_model(params)
            
            # Generate predictions
            y_pred = generate_predictions(model, X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Save metrics
            save_metrics(metrics, params)
            
            # Generate confusion matrix
            cm_path = plot_confusion_matrix(y_test, y_pred, params)
            
            # Log to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(params['evaluation']['metrics_path'])
            mlflow.log_artifact(cm_path)
            
            print("\n" + "=" * 60)
            print("✓ Evaluation stage completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error in evaluation stage: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
