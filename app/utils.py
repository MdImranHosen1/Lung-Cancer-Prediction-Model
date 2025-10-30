import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple, Any

# Global variables to store loaded models
model = None
scaler = None
encoders = None

def load_models():
    """Load the trained model, scaler, and encoders"""
    global model, scaler, encoders
    
    try:
        model = joblib.load("train_models/lung_cancer_model.pkl")
        scaler = joblib.load("train_models/scaler.pkl")
        encoders = joblib.load("train_models/encoders.pkl")
        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def predict_cancer(patient_data: Dict) -> Tuple[str, float]:
    """
    Make a prediction for a patient
    
    Args:
        patient_data: dict with raw values (e.g., 'M', 'F', 1, 2, actual age)
    
    Returns:
        tuple: (prediction, confidence)
    """
    if model is None or scaler is None or encoders is None:
        raise ValueError("Models are not loaded. Call load_models() first.")
    
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Encode only categorical columns using saved encoders
    for col in patient_df.columns:
        if col in encoders:  # Only encode if it's a categorical column
            patient_df[col] = encoders[col].transform(patient_df[col])
        # else: keep numeric values (like age) as-is
    
    # Scale
    patient_scaled = scaler.transform(patient_df)
    
    # Predict
    prediction = model.predict(patient_scaled)
    prediction_proba = model.predict_proba(patient_scaled)
    
    result = "Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"
    confidence = round(prediction_proba[0][prediction[0]] * 100, 2)
    
    return result, confidence

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj
def get_encoding_mappings():
    """Get encoding mappings for reference"""
    if encoders is None:
        return {}
    
    mappings = {}
    for col, encoder in encoders.items():
        if col != 'target':  # Skip target
            # Convert classes to native Python types before creating mapping
            classes = [convert_numpy_types(cls) for cls in encoder.classes_]
            transformed = encoder.transform(encoder.classes_)
            # Convert transformed values to native Python types
            transformed = [convert_numpy_types(val) for val in transformed]
            mapping = dict(zip(classes, transformed))
            mappings[col] = mapping
    return mappings

def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "Model not loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "features": getattr(model, 'n_features_in_', 'Unknown'),
        "status": "loaded"
    }
    return convert_numpy_types(info)