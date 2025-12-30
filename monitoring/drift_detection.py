"""
Model monitoring and drift detection (SIMPLE VERSION)
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime

def detect_data_drift(baseline_df, current_df, threshold=0.05):
    """Detect data drift using KS test"""
    drift_results = {}
    for col in baseline_df.columns:
        stat, pval = stats.ks_2samp(baseline_df[col], current_df[col])
        drift_results[col] = {
            "ks_statistic": float(stat),
            "p_value": float(pval),
            "drift_detected": pval < threshold  # This is the bool issue
        }
    return drift_results

def main():
    print("="*60)
    print("MODEL MONITORING & DRIFT DETECTION")
    print("="*60)
    
    # Load baseline data
    baseline_train = pd.read_csv("data/processed/X_train.csv")
    
    # Simulate production data with drift
    np.random.seed(42)
    n_samples = min(50, len(baseline_train))
    prod_data = baseline_train.sample(n=n_samples) + np.random.normal(0, 0.15, (n_samples, 4))
    prod_data = pd.DataFrame(prod_data, columns=baseline_train.columns)
    
    print(f"Baseline: {baseline_train.shape[0]} samples")
    print(f"Production: {prod_data.shape[0]} samples")
    
    # Detect drift
    baseline_sample = baseline_train.sample(n=n_samples)
    drift_report = detect_data_drift(baseline_sample, prod_data)
    
    # FIXED JSON SERIALIZATION (convert bool to str)
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_samples": int(baseline_train.shape[0]),
        "prod_samples": int(prod_data.shape[0]),
        "drift_detected": str(any(r["drift_detected"] for r in drift_report.values())),
        "drift_details": {
            k: {**v, "drift_detected": str(v["drift_detected"])} 
            for k, v in drift_report.items()
        },
        "features": list(baseline_train.columns)
    }
    
    # Save report
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/drift_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print(f"\n🚨 OVERALL: {'🟡 DRIFT DETECTED' if report['drift_detected'] == 'True' else '🟢 NORMAL'}")
    print("\n📊 FEATURE DRIFT:")
    for feature, result in drift_report.items():
        status = "🚨 DRIFT" if result["drift_detected"] else "✅ OK"
        print(f"  {status:8} {feature:<25} p={result['p_value']:.4f}")
    
    print(f"\n✓ Saved: outputs/drift_report.json")
    print("🎉 Monitoring COMPLETE!")

if __name__ == "__main__":
    main()
