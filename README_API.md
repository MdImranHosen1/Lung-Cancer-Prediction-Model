
# 🧬 Lung Cancer Prediction API

A **FastAPI-based** machine learning application for predicting **lung cancer risk** based on patient health and lifestyle data.
The API loads pre-trained models (using `joblib`), performs data preprocessing, encodes categorical features, scales inputs, and returns predictions with confidence scores.

---

## 🚀 Features

* Predicts **lung cancer likelihood** from patient data.
* Automatically loads ML model, scaler, and encoders at startup.
* Provides **confidence scores** with predictions.
* Offers **health** and **mapping** endpoints.
* Built with **FastAPI** for high performance.
* Fully supports **CORS** (cross-origin requests).
* Includes **Pydantic-based** input validation.

---

## 🗂️ Project Structure

```
lung_cancer_api/
│
├── app/
│   ├── main.py                
│   ├── models.py
│   ├── utils.py
│
├── train_models/
│   ├── lung_cancer_model.pkl    # Trained ML model
│   ├── scaler.pkl               # Feature scaler
│   ├── encoders.pkl             # Label encoders
│
└── run.py 
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MdImranHosen1/Lung-Cancer-Prediction-Model
cd lung-cancer-api
```

### 2. Create a Virtual Environment
**Windows:**

```bash
python -m venv lungenv
lungenv\Scripts\activate
```

**Linux / WSL:**

```bash
python -m venv lungenv
source lungenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
```

---

## ▶️ Running the API

Run the FastAPI app using:

```bash
python run.py
```

Or directly with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 API Endpoints

### **1️⃣ Root Endpoint**

**GET** `/`

```json
{
  "message": "Lung Cancer Prediction API",
  "status": "active",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "mappings": "/mappings"
  }
}
```

✅ Verifies that the API is running and lists available endpoints.

---

### **2️⃣ Health Check**

**GET** `/health`

**Response:**

```json
{
  "status": "healthy",
  "message": "API is running successfully"
}
```

✅ Confirms that the API and model are operational.

---

### **3️⃣ Encoding Mappings**

**GET** `/mappings`

Displays the reference mapping for encoded categorical values (e.g., `'M' -> 1`, `'F' -> 0`).

**Example Response:**

```json
{
  "status": "success",
  "mappings": {
    "gender": {"F": 0, "M": 1},
    "smoking": {"1": 0, "2": 1}
  }
}
```

---

### **4️⃣ Prediction**

**POST** `/predict`

Predicts whether a patient is at risk of lung cancer.

#### Request Body:

```json
{
  "gender": "M",
  "age": 65,
  "smoking": 2,
  "yellow_fingers": 2,
  "anxiety": 1,
  "peer_pressure": 2,
  "chronic_disease": 1,
  "fatigue": 2,
  "allergy": 1,
  "wheezing": 2,
  "alcohol_consuming": 1,
  "coughing": 2,
  "shortness_of_breath": 2,
  "swallowing_difficulty": 1,
  "chest_pain": 2
}
```

#### Response Example:

```json
{
  "prediction": "Lung Cancer",
  "confidence": 87.45,
  "status": "success",
  "message": "Prediction completed successfully"
}
```

#### Error Response:

```json
{
  "detail": "Prediction failed: Models are not loaded. Call load_models() first."
}
```

---

## 🧠 Model Details

### Model Files

All trained components are located in the `train_models/` directory:

* `lung_cancer_model.pkl` — the classifier model (e.g., RandomForest, LogisticRegression)
* `scaler.pkl` — the feature scaler used during training
* `encoders.pkl` — a dictionary of LabelEncoders for categorical columns

### Loading Process (`load_models()`)

At startup, FastAPI executes:

```python
@app.on_event("startup")
async def startup_event():
    load_models()
```

✅ Ensures that model, scaler, and encoders are loaded before serving predictions.

---

## 🧩 Key Modules

### **`app.models.py`**

Defines **Pydantic** data models for request validation and response formatting.

```python
class PatientData(BaseModel):
    gender: Literal['M', 'F']
    age: int
    smoking: int
    yellow_fingers: int
    ...
```

### **`app.utils.py`**

Handles:

* Model/scaler/encoder loading
* Prediction logic
* Data encoding and scaling
* JSON-safe type conversion

**Core Function:**

```python
def predict_cancer(patient_data: Dict) -> Tuple[str, float]:
    # Encode, scale, and predict
    prediction = model.predict(patient_scaled)
    prediction_proba = model.predict_proba(patient_scaled)
```

---

## 🧾 Example Workflow

1. **API starts**

   * Loads model, scaler, and encoders.
   * Prints available encodings.

2. **User sends POST request to `/predict`**

   * Input data is validated via Pydantic.
   * Encoders transform categorical fields.
   * Scaler standardizes numeric features.
   * Model predicts and returns label + confidence.

3. **Response returned**

   * Example: `"Lung Cancer"`, confidence: `92.35%`.

---

## 🧪 Testing the API (with cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "gender": "M",
  "age": 70,
  "smoking": 2,
  "yellow_fingers": 2,
  "anxiety": 1,
  "peer_pressure": 2,
  "chronic_disease": 1,
  "fatigue": 2,
  "allergy": 1,
  "wheezing": 2,
  "alcohol_consuming": 1,
  "coughing": 2,
  "shortness_of_breath": 2,
  "swallowing_difficulty": 1,
  "chest_pain": 2
}'
```

---

## 📊 Example Console Output on Startup

```
============================================================
LOADING MODEL AND ENCODERS
============================================================
✅ Models loaded successfully!

============================================================
ENCODING REFERENCE
============================================================
gender: {'F': 0, 'M': 1}
smoking: {'1': 0, '2': 1}
...
```

---

## 🧰 Utility Functions Overview

| Function                       | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `load_models()`                | Loads `model`, `scaler`, and `encoders` from disk.   |
| `predict_cancer(patient_data)` | Encodes, scales, and predicts cancer risk.           |
| `convert_numpy_types(obj)`     | Converts NumPy types to JSON-safe Python types.      |
| `get_encoding_mappings()`      | Returns categorical encoding mappings for reference. |
| `get_model_info()`             | Returns details about the loaded ML model.           |

---

## 🧾 Example Output from `get_model_info()`

```json
{
  "model_type": "RandomForestClassifier",
  "features": 15,
  "status": "loaded"
}
```

---

## 🛠️ Future Enhancements

* Add **model retraining API**.
* Implement **database logging** for predictions.
* Add **Swagger documentation examples**.
* Secure endpoints using API keys or JWT.
* Enable **Docker deployment**.

---

