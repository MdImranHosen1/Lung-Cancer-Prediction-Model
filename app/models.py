from pydantic import BaseModel, Field
from typing import Literal

class PatientData(BaseModel):
    gender: Literal['M', 'F'] = Field(..., description="Gender: M or F")
    age: int = Field(..., ge=1, le=120, description="Age between 1 and 120")
    smoking: int = Field(..., ge=1, le=2, description="Smoking: 1 (No) or 2 (Yes)")
    yellow_fingers: int = Field(..., ge=1, le=2, description="Yellow fingers: 1 (No) or 2 (Yes)")
    anxiety: int = Field(..., ge=1, le=2, description="Anxiety: 1 (No) or 2 (Yes)")
    peer_pressure: int = Field(..., ge=1, le=2, description="Peer pressure: 1 (No) or 2 (Yes)")
    chronic_disease: int = Field(..., ge=1, le=2, description="Chronic disease: 1 (No) or 2 (Yes)")
    fatigue: int = Field(..., ge=1, le=2, description="Fatigue: 1 (No) or 2 (Yes)")
    allergy: int = Field(..., ge=1, le=2, description="Allergy: 1 (No) or 2 (Yes)")
    wheezing: int = Field(..., ge=1, le=2, description="Wheezing: 1 (No) or 2 (Yes)")
    alcohol_consuming: int = Field(..., ge=1, le=2, description="Alcohol consuming: 1 (No) or 2 (Yes)")
    coughing: int = Field(..., ge=1, le=2, description="Coughing: 1 (No) or 2 (Yes)")
    shortness_of_breath: int = Field(..., ge=1, le=2, description="Shortness of breath: 1 (No) or 2 (Yes)")
    swallowing_difficulty: int = Field(..., ge=1, le=2, description="Swallowing difficulty: 1 (No) or 2 (Yes)")
    chest_pain: int = Field(..., ge=1, le=2, description="Chest pain: 1 (No) or 2 (Yes)")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    status: str
    message: str