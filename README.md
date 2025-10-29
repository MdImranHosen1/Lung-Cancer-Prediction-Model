
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
│   └── confusion_matrix.png
└── lung_cancer_train.py
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
