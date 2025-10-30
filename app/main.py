from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import PatientData, PredictionResponse
from app.utils import load_models, predict_cancer, get_encoding_mappings
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API for predicting lung cancer risk based on patient data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    print("="*60)
    print("LOADING MODEL AND ENCODERS")
    print("="*60)
    
    if not load_models():
        raise Exception("Failed to load models during startup")
    
    # Print encoding mappings
    mappings = get_encoding_mappings()
    print("\n" + "="*60)
    print("ENCODING REFERENCE")
    print("="*60)
    for col, mapping in mappings.items():
        print(f"{col}: {mapping}")

@app.get("/")
async def root():
    return {
        "message": "Lung Cancer Prediction API", 
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "mappings": "/mappings"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running successfully"
    }

@app.get("/mappings")
async def get_mappings():
    """Get encoding mappings for reference"""
    mappings = get_encoding_mappings()
    return {
        "status": "success",
        "mappings": mappings
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Make lung cancer prediction based on patient data
    
    - **gender**: M or F
    - **age**: Patient's age (1-120)
    - All other fields: 1 (No) or 2 (Yes)
    """
    try:
        # Convert Pydantic model to dict
        data_dict = patient_data.dict()
        
        # Make prediction
        prediction, confidence = predict_cancer(data_dict)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            status="success",
            message="Prediction completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

