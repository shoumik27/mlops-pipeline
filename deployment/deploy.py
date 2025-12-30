"""
Production ML Model API Server
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
SCALER_PATH = "data/processed/scaler.pkl"

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading scaler from: {SCALER_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions"""
    if not model or not scaler:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400
        
        # Validate features length (Iris has 4 features)
        features = np.array(data["features"]).reshape(1, -1)
        if features.shape[1] != 4:
            return jsonify({"error": "Expected 4 features (Iris dataset)"}), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0].tolist()
        
        response = {
            "prediction": int(prediction),
            "probabilities": probabilities,
            "features": data["features"],
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model_info", methods=["GET"])
def model_info():
    """Model metadata"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": str(type(model).__name__),
        "n_features": getattr(model, 'n_features_in_', 'Unknown'),
        "n_classes": getattr(model, 'n_classes_', 'Unknown'),
        "model_path": MODEL_PATH
    })

@app.route("/", methods=["GET"])
def root():
    """API documentation"""
    return jsonify({
        "message": "ML Model API",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST JSON with 'features')",
            "/model_info": "Model metadata"
        },
        "example_predict": {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    })

if __name__ == "__main__":
    print("🚀 Starting ML Model API Server...")
    print(f"📊 Model: {MODEL_PATH}")
    print(f"📈 Scaler: {SCALER_PATH}")
    print("🔗 Health: http://localhost:5000/health")
    print("🔗 Predict: http://localhost:5000/predict")
    app.run(host="0.0.0.0", port=5000, debug=False)
