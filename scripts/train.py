"""
Model training with MLflow experiment tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

def load_processed_data():
    """Load preprocessed data"""
    print("Loading processed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    print(f"Training data shape: {X_train.shape}")
    return X_train, y_train

def train_model(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train Random Forest model"""
    print(f"\nTraining Random Forest with n_estimators={n_estimators}, max_depth={max_depth}...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    
    return model, {"accuracy": train_acc}

def save_model(model, model_path="models/model.pkl"):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

def main():
    """Main training pipeline"""
    print("="*70)
    print("MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Start MLflow experiment
    mlflow.set_experiment("iris-classification")
    
    with mlflow.start_run(run_name="random_forest_baseline") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")
        
        # Load data
        X_train, y_train = load_processed_data()
        
        # Set parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "model_type": "RandomForest"
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model, metrics = train_model(
            X_train, y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save locally
        save_model(model)
        
        # Save metrics to file
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "="*70)
        print("TRAINING METRICS")
        print("="*70)
        print(json.dumps(metrics, indent=2))
        
        print(f"\n✓ MLflow Tracking Server: http://localhost:5000")
        print(f"✓ Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
