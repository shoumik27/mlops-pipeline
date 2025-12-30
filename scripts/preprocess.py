"""
Data preprocessing pipeline with DVC tracking
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os

def load_data():
    """Load iris dataset"""
    print("Loading data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocess data"""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train, X_test, y_train, y_test, scaler, output_dir="data/processed"):
    """Save processed data"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving data to {output_dir}...")
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    print("✓ Data saved successfully")
    
    # Log data statistics
    stats = {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "features": list(X_train.columns),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict()
    }
    
    with open(f"{output_dir}/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats

def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load data
    X, y = load_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Save
    stats = save_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\n" + "="*70)
    print("DATA STATISTICS")
    print("="*70)
    print(json.dumps(stats, indent=2, default=str))
    
    print("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    main()
