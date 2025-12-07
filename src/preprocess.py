"""
Data Preprocessing Stage
Performs feature scaling, train/test split, and saves processed data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import sys


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_clean_data(params):
    """
    Load clean data from interim directory.
    
    Args:
        params (dict): Configuration parameters
        
    Returns:
        pd.DataFrame: Clean dataset
    """
    interim_path = params['data']['interim']
    print(f"Loading clean data from {interim_path}...")
    
    df = pd.read_csv(interim_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    return df


def scale_features(df):
    """
    Apply StandardScaler to 'Time' and 'Amount' features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with scaled features
    """
    print("\nScaling 'Time' and 'Amount' features...")
    
    scaler = StandardScaler()
    
    # Scale Time and Amount columns
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    print("Feature scaling completed!")
    return df


def split_data(df, params):
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        params (dict): Configuration parameters
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as numpy arrays
    """
    print("\nSplitting data into train and test sets...")
    
    # Separate features and target
    X = df.drop(columns='Class', axis=1)
    y = df['Class']
    
    # Get preprocessing parameters
    test_size = params['preprocessing']['test_size']
    random_state = params['preprocessing']['random_state']
    stratify = y if params['preprocessing']['stratify'] else None
    
    # Perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    print(f"  - Train class distribution:\n{y_train.value_counts()}")
    print(f"  - Test class distribution:\n{y_test.value_counts()}")
    
    # Convert to numpy arrays
    return X_train.values, X_test.values, y_train.values, y_test.values


def save_processed_data(X_train, X_test, y_train, y_test, params):
    """
    Save processed data as numpy arrays.
    
    Args:
        X_train, X_test, y_train, y_test: Numpy arrays
        params (dict): Configuration parameters
    """
    print("\nSaving processed data...")
    
    processed_paths = params['data']['processed']
    
    np.save(processed_paths['X_train'], X_train)
    np.save(processed_paths['X_test'], X_test)
    np.save(processed_paths['y_train'], y_train)
    np.save(processed_paths['y_test'], y_test)
    
    print(f"  - Saved {processed_paths['X_train']}")
    print(f"  - Saved {processed_paths['X_test']}")
    print(f"  - Saved {processed_paths['y_train']}")
    print(f"  - Saved {processed_paths['y_test']}")
    print("Processed data saved successfully!")


def main():
    """Main execution function for preprocessing stage."""
    try:
        # Load parameters
        params = load_params()
        
        # Load clean data
        df = load_clean_data(params)
        
        # Scale features
        df = scale_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df, params)
        
        # Save processed data
        save_processed_data(X_train, X_test, y_train, y_test, params)
        
        print("\n✓ Preprocessing stage completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in preprocessing stage: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
