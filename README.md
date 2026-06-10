# ❤️ CardioPredict — Heart Failure Risk Predictor
### Machine Learning Edition v2.0 | Logistic Regression | Flask API

---

## 📌 Project Overview

**CardioPredict** is a full-stack web application that predicts heart failure risk using a trained **Logistic Regression** machine learning model. It collects 12 clinical health parameters from the user, sends them to a **Flask REST API** backend, and returns a real-time risk prediction with confidence score and personalised medical advice.

> ⚠️ **Medical Disclaimer:** This tool is for educational and academic purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **ML Algorithm** | Logistic Regression (scikit-learn) |
| **Backend** | Python 3.x + Flask |
| **Data Processing** | NumPy, Pandas |
| **Model Persistence** | pickle |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **API Communication** | REST API (JSON) + fetch() |
| **Styling** | Glassmorphism Dark Theme |
| **Mobile Packaging** | Apache Cordova |

---

## 📁 Project Structure

```
Cardiopredict Heart failure detection/
│
├── 🐍 generate_dataset.py     # Generates synthetic heart failure dataset (1500 records)
├── 🧠 train_model.py          # Trains Logistic Regression, saves artifacts
├── 🌐 app.py                  # Flask REST API backend (serves app + predictions)
│
├── 📄 index.html              # Main prediction dashboard (MODIFIED for ML)
├── 📄 login.html              # Login page (unchanged)
├── 📄 change-password.html    # Settings page (unchanged)
├── 🎨 style.css               # Dark glassmorphism UI (unchanged)
├── ⚙️  script.js              # Frontend logic — now calls Flask API (MODIFIED)
│
├── 📦 requirements.txt        # Python dependencies
├── 🔧 setup.bat               # One-click Windows setup script
│
├── heart_failure_dataset.csv  # Generated training data (after running generate_dataset.py)
├── model.pkl                  # Trained Logistic Regression model (after training)
├── scaler.pkl                 # Fitted StandardScaler (after training)
└── model_info.json            # Model performance metrics (after training)
```

---

## ⚡ Quick Setup (Windows)

### Option A — One-Click Setup
```
Double-click setup.bat
```

### Option B — Manual Steps

**Step 1:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 2:** Generate the training dataset
```bash
python generate_dataset.py
```

**Step 3:** Train the Logistic Regression model
```bash
python train_model.py
```

**Step 4:** Start the Flask server
```bash
python app.py
```

**Step 5:** Open the app in your browser
```
http://localhost:5000
```
> Default login: **admin** / **admin123**

---

## 🔁 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      BROWSER (Frontend)                      │
│                                                              │
│  login.html  ──►  index.html                                 │
│                      │                                       │
│                   script.js                                  │
│                      │  POST /predict (JSON)                 │
└──────────────────────┼──────────────────────────────────────┘
                        │
                        ▼ HTTP (localhost:5000)
┌─────────────────────────────────────────────────────────────┐
│                    Flask API (app.py)                        │
│                                                              │
│  1. Receive JSON  ──►  validate_input()                      │
│  2. encode_input()  →  [age, gender, bp, chol, ...]          │
│  3. scaler.transform()  →  normalised feature vector         │
│  4. model.predict_proba()  →  P(heart failure)               │
│  5. extract_risk_factors()  →  interpretable factors         │
│  6. Return JSON response                                     │
└──────────────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   model.pkl                      scaler.pkl
 (LogisticRegression)          (StandardScaler)
         ▲                             ▲
         └──────── train_model.py ─────┘
                        ▲
              heart_failure_dataset.csv
                        ▲
               generate_dataset.py
```

---

## 🧠 Machine Learning Pipeline

### 1. Dataset Generation (`generate_dataset.py`)
- Generates **1500 synthetic patient records** with clinically realistic distributions
- 12 input features mirroring the web form fields
- Binary target: `heart_failure` (0 = No Risk, 1 = Risk)
- ~42% positive class (heart failure risk)

### 2. Feature Engineering
| Feature | Type | Encoding |
|---|---|---|
| age | Continuous | As-is |
| gender | Categorical | male=1, female=0 |
| systolic_bp | Continuous | As-is (mmHg) |
| diastolic_bp | Continuous | As-is (mmHg) |
| cholesterol | Continuous | As-is (mg/dL) |
| blood_sugar | Continuous | As-is (mg/dL) |
| bmi | Continuous | As-is (kg/m²) |
| activity_level | Ordinal | sedentary=0 … very-active=4 |
| diabetes | Ordinal | no=0, prediabetes=1, yes=2 |
| smoking | Ordinal | never=0, former=1, current=2 |
| family_history | Ordinal | no=0, unknown=1, yes=2 |
| previous_heart_issues | Binary | no=0, yes=1 |

### 3. Preprocessing
- **StandardScaler**: Normalises features to mean=0, std=1
- Fit on training data only (prevents data leakage)
- Applied to test data using training statistics

### 4. Model — Logistic Regression
```
sigmoid(z) = 1 / (1 + e^(-z))
z = w₀ + w₁·age + w₂·gender + ... + w₁₂·previous_heart
```
- **Penalty**: L2 (Ridge) regularisation
- **Solver**: lbfgs
- **Class weight**: balanced
- **Max iterations**: 1000

### 5. Evaluation
| Metric | Description |
|---|---|
| **Accuracy** | % of correct predictions |
| **Precision** | Of predicted positives, how many are correct |
| **Recall** | Of actual positives, how many were found |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC-ROC** | Area under ROC curve (discrimination ability) |
| **5-fold CV** | Cross-validation to detect overfitting |

---

## 🌐 API Reference

### `POST /predict`
Run ML prediction on patient data.

**Request:**
```json
{
  "age": 58,
  "gender": "male",
  "systolic": 145,
  "diastolic": 92,
  "cholesterol": 235,
  "bloodSugar": 118,
  "bmi": 29.4,
  "activity": "light",
  "diabetes": "prediabetes",
  "smoking": "former",
  "familyHistory": "yes",
  "previousHeart": "no"
}
```

**Response:**
```json
{
  "success": true,
  "risk_percentage": 71.3,
  "risk_level": "High Risk",
  "level_class": "risk-high",
  "confidence": 42.6,
  "probability": 0.713,
  "prediction": 1,
  "risk_factors": [
    { "factor": "Age over 50", "severity": "moderate" },
    { "factor": "Stage 2 Hypertension (145/92 mmHg)", "severity": "high" }
  ],
  "model_accuracy": 87.33,
  "auc_score": 93.10,
  "algorithm": "Logistic Regression"
}
```

### `GET /model-info`
Returns model performance metrics for the UI.

### `GET /health`
Returns `{ "status": "running", "model_loaded": true }`.

---

## 🔐 Authentication

- Credentials stored in browser `localStorage`
- Session tracked via `sessionStorage` (`isLoggedIn`)
- Default: **admin / admin123**
- Change via ⚙️ Settings → Change Password

---

## 🎓 Viva Q&A Guide

**Q: Why Logistic Regression for this problem?**
> Heart failure risk is a binary classification problem (risk / no risk). Logistic Regression is interpretable, computationally efficient, and gives probability outputs — ideal for medical risk scoring where explainability matters.

**Q: What is the sigmoid function?**
> `σ(z) = 1 / (1 + e^(-z))`. It maps any real number to (0, 1), giving us a probability for the positive class.

**Q: Why StandardScaler?**
> Logistic Regression uses gradient descent. Features on very different scales (age: 18–95 vs cholesterol: 100–400) cause slow convergence and biased coefficients. Scaling ensures all features contribute equally.

**Q: What is L2 regularisation?**
> Adds a penalty term `λ·Σwᵢ²` to the loss function to prevent overfitting by discouraging large weights.

**Q: Why stratified train/test split?**
> Ensures both training (80%) and test (20%) sets maintain the same positive/negative class ratio, preventing biased evaluation.

**Q: What does AUC-ROC measure?**
> The probability that the model ranks a randomly chosen positive case higher than a randomly chosen negative case. AUC = 1.0 is perfect; 0.5 is random guessing.

**Q: What is the Confusion Matrix?**
> A 2×2 table: TN (correct negatives), FP (false alarms), FN (missed cases), TP (correct positives). Used to calculate Precision, Recall, and F1.

**Q: How does the frontend connect to the model?**
> The JavaScript `fetch()` API sends a POST request with form data as JSON to the Flask `/predict` endpoint. Flask encodes, scales, and passes data through the pickled Logistic Regression model, returning a JSON prediction response.

---

## 👤 Login Credentials

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `admin123` |

---

*CardioPredict v2.0 — Final Year Project | Machine Learning for Healthcare*
