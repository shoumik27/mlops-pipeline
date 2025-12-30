"""
Model evaluation with comprehensive metrics
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_and_model():
    """Load test data and trained model"""
    print("Loading test data...")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    print("Loading trained model...")
    model = joblib.load("models/model.pkl")
    
    print(f"Test data shape: {X_test.shape}")
    return X_test, y_test, model

def evaluate_model(X_test, y_test, model):
    """Evaluate model on test data"""
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\nEvaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    return metrics, y_pred, report

def create_confusion_matrix_plot(y_test, y_pred, save_path="outputs/confusion_matrix.png"):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")

def create_metrics_plot(metrics, save_path="outputs/metrics_bar.png"):
    """Create metrics bar plot"""
    plt.figure(figsize=(10, 5))
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    
    plt.bar(metrics_names, metrics_values, color='steelblue', alpha=0.8, edgecolor='black')
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics bar plot saved to {save_path}")

def main():
    """Main evaluation pipeline"""
    print("="*70)
    print("MODEL EVALUATION PIPELINE")
    print("="*70)
    
    # Start MLflow run
    mlflow.set_experiment("iris-classification")
    
    with mlflow.start_run(run_name="model_evaluation") as run:
        # Load data and model
        X_test, y_test, model = load_data_and_model()
        
        # Evaluate
        metrics, y_pred, report = evaluate_model(X_test, y_test, model)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Create visualizations
        create_confusion_matrix_plot(y_test, y_pred)
        create_metrics_plot(metrics)
        
        # Log artifacts
        mlflow.log_artifact("outputs/confusion_matrix.png")
        mlflow.log_artifact("outputs/metrics_bar.png")
        
        # Save metrics to file
        with open("outputs/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifacts saved to outputs/")
        print(f"Check MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    main()
