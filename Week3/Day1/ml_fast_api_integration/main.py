import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Day 14: ML & FastAPI Capstone Integration",
    description="Serves a trained Random Forest classifier pipeline predicting breast cancer diagnosis (Malignant vs Benign).",
    version="1.0.0"
)

# --- Path to Saved Pipeline ---
# Use path relative to the Day 14 subfolder or absolute path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "cancer_prediction_pipeline.joblib")

# Load the model once on startup
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Successfully loaded trained model pipeline from {MODEL_PATH}")
else:
    model_pipeline = None
    print(f"Warning: Model file not found at {MODEL_PATH}. Run train.py first to generate it!")

# --- Schemas ---

class PredictionInput(BaseModel):
    # Expect 30 floats matching the breast cancer dataset features
    features: List[float] = Field(
        ..., 
        min_items=30, 
        max_items=30, 
        description="A list containing exactly 30 numerical features corresponding to tumor measurements."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }
        }

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Predicted class: 0 (Malignant) or 1 (Benign)")
    prediction_label: str = Field(..., description="Diagnosis label ('malignant' or 'benign')")
    probability: float = Field(..., description="Confidence probability of the predicted class")

# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "message": "Day 14 ML Model Deployment Service",
        "model_loaded": model_pipeline is not None,
        "swagger_docs": "/docs"
    }

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(payload: PredictionInput):
    """
    Predicts breast cancer diagnosis using the incoming 30 features.
    
    The API:
    1. Validates the feature list length (must be exactly 30).
    2. Passes the features to the loaded StandardScaler + RandomForest pipeline.
    3. Returns the classification result and confidence level.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model is not available. Please run train.py first."
        )
        
    try:
        # Convert list to 2D numpy array (shape: 1, 30)
        input_data = np.array(payload.features).reshape(1, -1)
        
        # Predict class
        pred_class = int(model_pipeline.predict(input_data)[0])
        
        # Predict class probabilities
        pred_probs = model_pipeline.predict_proba(input_data)[0]
        confidence = float(pred_probs[pred_class])
        
        # Mapping target labels (0: malignant, 1: benign in Breast Cancer dataset)
        labels = {0: "malignant", 1: "benign"}
        label = labels.get(pred_class, "unknown")
        
        return PredictionOutput(
            prediction=pred_class,
            prediction_label=label,
            probability=confidence
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Start capstone service
    uvicorn.run(app, host="127.0.0.1", port=8000)
