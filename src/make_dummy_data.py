"""
Script to generate dummy data for CI/CD when DVC remote is inaccessible.
"""
import pandas as pd
import numpy as np
import os
import yaml

def generate_dummy_data():
    print("Generating dummy credit card data for CI/CD...")
    
    # Load params to check path
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    raw_path = params['data']['raw']
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    # Generate 1000 samples (enough for train/test split)
    n_samples = 1000
    
    # Features: Time, V1...V28, Amount
    data = {
        'Time': np.arange(n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples)
    }
    
    # Add V1 through V28
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
    # Add Class (mostly 0, some 1)
    # Ensure at least a few fraud cases for stratified split
    y = np.zeros(n_samples, dtype=int)
    y[:10] = 1  # 1% fraud
    np.random.shuffle(y)
    data['Class'] = y
    
    df = pd.DataFrame(data)
    
    # Save to raw path
    df.to_csv(raw_path, index=False)
    print(f"Dummy data saved to {raw_path}")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

if __name__ == "__main__":
    generate_dummy_data()
