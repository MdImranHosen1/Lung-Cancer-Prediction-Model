
---

# 🫁 Lung Cancer Prediction Model

A machine learning project that predicts **lung cancer risk** using survey data.

---

## ⚙️ Setup Instructions

### 1️⃣ Create and Activate Virtual Environment

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

### 2️⃣ Install Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 📊 Dataset

**Source:**
👉 [Kaggle - Lung Cancer Survey Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer?select=survey+lung+cancer.csv)

**File used:**
`dataset/survey_lung_cancer.csv`

---

## 📘 Reference

📄 Related notebook:
[Lung Cancer Analysis (Accuracy 96.4%)](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4)

---

## ⚙️ Features
- Automatic data cleaning and encoding
- Compares 7 ML Models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - Neural Network (MLP)
    - Decision Tree
    - K-Nearest Neighbors (KNN)
- Saves:
    - ✅ Best model
    - ✅ Scaler
    - ✅ Encoders
    - ✅ Categorical column info
    - ✅ Model comparison chart + CSV summary

---

## 🚀 Run Training

```
python train_lung_cancer_model.py
```

### Output Example:
```
Model                     Accuracy     CV Score     ROC-AUC     
----------------------------------------------------------------------
Logistic Regression       0.9103       0.9349       0.9500      
Random Forest             0.9231       0.9133       0.9610      
Gradient Boosting         0.8718       0.8787       0.9412      
SVM                       0.8846       0.9046       0.9426      
Neural Network            0.8974       0.9090       0.8735      
Decision Tree             0.8974       0.8875       0.8559      
KNN                       0.8974       0.9090       0.9206      
======================================================================

🏆 BEST MODEL: Random Forest
✅ Accuracy: 0.9231 (92.31%)
```

---

## 🪜 Project Steps

1️⃣ **Load & Clean Data**

* Standardize column names
* Rename `lung_cancer → target`

2️⃣ **Encode Categorical Columns**

* Use `LabelEncoder` for gender, smoking, etc.

3️⃣ **Split Dataset**

* 75% train, 25% test (`train_test_split`)

4️⃣ **Scale Features**

* Normalize data using `StandardScaler`

5️⃣ **Train Model**

* Logistic Regression (`max_iter=1000`)

6️⃣ **Evaluate & Save**

* Accuracy, confusion matrix
* Save model, scaler, encoders with `joblib`

---

## 🧠 Prediction Flow

```
New Patient → Encode → Scale → Model → Prediction
```

---

## 🧩 Example Prediction

### First example
```python
result, confidence = predict_cancer({
    'gender': 'M',
    'age': 65,
    'smoking': 1,
    'yellow_fingers': 2,
    'anxiety': 2,
    'peer_pressure': 1,
    'chronic_disease': 1,
    'fatigue': 2,
    'allergy': 1,
    'wheezing': 2,
    'alcohol_consuming': 2,
    'coughing': 2,
    'shortness_of_breath': 2,
    'swallowing_difficulty': 2,
    'chest_pain': 2
})
print(result, confidence)
```
### Output
```
🩺 TEST 1: High-Risk Patient
---------------------------------------
🔮 Prediction: Lung Cancer
🧩 Confidence: 100.0%
```
---
### Second example
```python
# Example 3: Another NO case (F,63,1,2,1,1,1,1,1,2,1,2,2,1,1,NO)
print("\n🩺 TEST 3: Another Low-Risk Patient")
print("-" * 60)
another_no = {
    'gender': 'F',
    'age': 63,
    'smoking': 1,
    'yellow_fingers': 2,
    'anxiety': 1,
    'peer_pressure': 1,
    'chronic_disease': 1,
    'fatigue': 1,
    'allergy': 1,
    'wheezing': 2,
    'alcohol_consuming': 1,
    'coughing': 2,
    'shortness_of_breath': 2,
    'swallowing_difficulty': 1,
    'chest_pain': 1
}

result, confidence = predict_cancer(another_no)
print("Patient Data:", another_no)
print(f"🔮 Prediction: {result}")
print(f"🧩 Confidence: {confidence}%")
```
### Output
```
🩺 TEST 3: Another Low-Risk Patient
---------------------------------------
🔮 Prediction: No Lung Cancer
🧩 Confidence: 92.0%
```
---

## 📁 Folder Structure

```
project/
├── dataset/
│   └── survey_lung_cancer.csv
├── train_models/
│   ├── lung_cancer_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── categorical_columns.pkl
│   ├── best_model_name.pkl
│   ├── model_comparison.png
│   └── model_comparison_results.csv
├── lung_cancer_train_model.ipynb
└── lung_cancer_predict.ipynb
```

---

## 💾 Saved Files

| File                    | Purpose                    |
| ----------------------- | -------------------------- |
| `lung_cancer_model.pkl` | Trained ML model           |
| `scaler.pkl`            | Normalizes new input data  |
| `encoders.pkl`          | Encodes categorical values |
| `confusion_matrix.png`  | Evaluation result          |

---

## ✅ Summary

* Predicts lung cancer using survey data
* Includes full training → evaluation → prediction flow
* Uses Logistic Regression for classification
* Ready to extend with more ML models
