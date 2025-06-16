import os
import joblib

MODEL_PATH = "./models/XGBoost.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

# Load the model
model = load_model()
