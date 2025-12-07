"""
Data Loading Stage
Loads raw credit card fraud dataset and performs basic validation.
"""

import pandas as pd
import yaml
import sys


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_data(params):
    """
    Load raw credit card fraud dataset.
    
    Args:
        params (dict): Configuration parameters
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    raw_data_path = params['data']['raw']
    print(f"Loading data from {raw_data_path}...")
    
    df = pd.read_csv(raw_data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    return df


def validate_data(df):
    """
    Perform basic data validation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        bool: True if validation passes
    """
    print("\nData Validation:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Null values: {df.isnull().sum().sum()}")
    print(f"  - Class distribution:\n{df['Class'].value_counts()}")
    
    # Check for required columns
    required_columns = ['Class', 'Time', 'Amount']
    for col in required_columns:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            return False
    
    # Check expected number of features (30 features + 1 target)
    if df.shape[1] != 31:
        print(f"WARNING: Expected 31 columns, got {df.shape[1]}")
    
    print("Data validation passed!")
    return True


def save_clean_data(df, params):
    """
    Save cleaned data to interim directory.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        params (dict): Configuration parameters
    """
    interim_path = params['data']['interim']
    print(f"\nSaving clean data to {interim_path}...")
    
    df.to_csv(interim_path, index=False)
    print("Clean data saved successfully!")


def main():
    """Main execution function for data loading stage."""
    try:
        # Load parameters
        params = load_params()
        
        # Load data
        df = load_data(params)
        
        # Validate data
        if not validate_data(df):
            sys.exit(1)
        
        # Save clean data
        save_clean_data(df, params)
        
        print("\n✓ Data loading stage completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in data loading stage: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
