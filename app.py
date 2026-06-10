"""
============================================================
  CardioPredict — Flask REST API Backend  (v2.0)
============================================================
This Flask application serves as the backend server for the
CardioPredict Heart Failure Risk Prediction web application.

It loads the trained Logistic Regression model and provides
REST API endpoints consumed by the HTML/JS frontend.

Endpoints:
  GET  /                    -> Serve login page (entry point)
  GET  /index.html          -> Serve main dashboard
  GET  /change-password.html-> Serve change-password page
  GET  /<filename>          -> Serve any static file (css, js)
  POST /predict             -> Run ML prediction (main endpoint)
  GET  /model-info          -> Return model performance metrics
  GET  /health              -> Health check

Run:  python app.py
Open: http://localhost:5000
Default Login: admin / admin123
============================================================
"""

import os
import pickle
import json
import warnings

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings('ignore')


# ============================================================
# FLASK APP SETUP
# Static folder = current directory so Flask serves all HTML/CSS/JS
# ============================================================
app = Flask(__name__, static_folder='.', static_url_path='')

# Allow cross-origin requests from any origin
# (needed when opening HTML files directly from filesystem during dev)
CORS(app, resources={r"/*": {"origins": "*"}})


# ============================================================
# FEATURE COLUMNS — must match generate_dataset.py & train_model.py
# ============================================================
FEATURE_COLUMNS = [
    'age',
    'gender',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'cholesterol',
    'blood_sugar',
    'diabetes',
    'family_history',
    'previous_heart_problems',
    'smoking',
    'physical_activity'
]


# ============================================================
# LOAD MODEL ARTIFACTS ON STARTUP
# ============================================================
print("\n  Loading ML model artifacts...")

MODEL_LOADED = False
model        = None
scaler       = None
model_info   = {}

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("  ✅ model.pkl loaded")

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("  ✅ scaler.pkl loaded")

    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"  ✅ model_info.json loaded  "
          f"(Accuracy: {model_info['accuracy']*100:.2f}%, "
          f"AUC: {model_info['auc_score']*100:.2f}%)")

    MODEL_LOADED = True

except FileNotFoundError as e:
    print(f"  ❌ {e}")
    print("     Please run:  python train_model.py  (and generate_dataset.py first)")


# ============================================================
# HELPER: INPUT ENCODING
# Converts string values from the HTML form to numeric values
# that match the training dataset encoding.
# ============================================================

def encode_input(data: dict) -> np.ndarray:
    """
    Convert the HTML form JSON payload into a (1, 12) numeric feature vector
    suitable for scaler.transform() and model.predict().

    Args:
        data (dict): Raw JSON from the frontend form.

    Returns:
        np.ndarray: Shape (1, 12) — one row, 12 features.
    """
    # 1. Age (continuous)
    age = float(data.get('age', 0))
    
    # 2. Gender (1=Female, 2=Male based on dataset stats)
    gender_str = data.get('gender', 'other')
    gender = 2.0 if gender_str == 'male' else 1.0
    
    # 3. BMI (continuous)
    bmi = float(data.get('bmi', 25.0))
    
    # 4. Systolic BP (continuous)
    systolic_bp = float(data.get('systolic', 120))
    
    # 5. Diastolic BP (continuous)
    diastolic_bp = float(data.get('diastolic', 80))
    
    # 6. Cholesterol (1 = Normal < 200, 2 = Borderline 200-239, 3 = High >= 240)
    chol_val = float(data.get('cholesterol', 200))
    if chol_val < 200:
        cholesterol = 1.0
    elif chol_val < 240:
        cholesterol = 2.0
    else:
        cholesterol = 3.0
        
    # 7. Blood Sugar (1 = Normal < 100, 2 = Pre-diabetic 100-125, 3 = Diabetic >= 126)
    bs_val = float(data.get('bloodSugar', 100))
    if bs_val < 100:
        blood_sugar = 1.0
    elif bs_val < 126:
        blood_sugar = 2.0
    else:
        blood_sugar = 3.0
        
    # 8. Diabetes (0 = No, 1 = Yes)
    diabetes_str = data.get('diabetes', 'no')
    diabetes = 1.0 if diabetes_str == 'yes' else 0.0
    
    # 9. Family History (0 = No, 1 = Yes)
    family_str = data.get('familyHistory', 'no')
    family_history = 1.0 if family_str == 'yes' else 0.0
    
    # 10. Previous Heart Problems (0 = No, 1 = Yes)
    prev_str = data.get('previousHeart', 'no')
    previous_heart_problems = 1.0 if prev_str == 'yes' else 0.0
    
    # 11. Smoking (0 = No, 1 = Yes)
    smoking_str = data.get('smoking', 'never')
    smoking = 1.0 if smoking_str == 'current' else 0.0
    
    # 12. Physical Activity (0 = Inactive/Sedentary, 1 = Active)
    activity_str = data.get('activity', 'moderate')
    physical_activity = 0.0 if activity_str == 'sedentary' else 1.0

    feature_vector = [
        age,
        gender,
        bmi,
        systolic_bp,
        diastolic_bp,
        cholesterol,
        blood_sugar,
        diabetes,
        family_history,
        previous_heart_problems,
        smoking,
        physical_activity
    ]
    return np.array(feature_vector).reshape(1, -1)


def validate_input(data: dict) -> tuple:
    """
    Validate all required fields and numeric ranges.

    Args:
        data (dict): Request JSON payload.

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    required = ['age', 'gender', 'systolic', 'diastolic',
                'cholesterol', 'bloodSugar', 'bmi', 'activity',
                'diabetes', 'smoking', 'familyHistory', 'previousHeart']

    # Check all required fields are present
    for field in required:
        if field not in data or data[field] == '' or data[field] is None:
            return False, f"Missing required field: '{field}'"

    # Validate numeric ranges
    checks = [
        ('age',         int,   data['age'],         1,    120, "Age must be 1–120 years"),
        ('systolic',    int,   data['systolic'],    70,    250, "Systolic BP must be 70–250 mmHg"),
        ('diastolic',   int,   data['diastolic'],   40,    150, "Diastolic BP must be 40–150 mmHg"),
        ('cholesterol', int,   data['cholesterol'], 100,   400, "Cholesterol must be 100–400 mg/dL"),
        ('bloodSugar',  int,   data['bloodSugar'],   50,   400, "Blood sugar must be 50–400 mg/dL"),
        ('bmi',         float, data['bmi'],          10.0, 60.0,"BMI must be 10–60 kg/m²"),
    ]

    for field, cast_fn, value, lo, hi, msg in checks:
        try:
            v = cast_fn(value)
            if not (lo <= v <= hi):
                return False, msg
        except (ValueError, TypeError):
            return False, f"Invalid numeric value for '{field}'"

    return True, ""


def extract_risk_factors(data: dict) -> list:
    """
    Identify and explain the patient's individual risk factors
    for display in the frontend results panel.

    Args:
        data (dict): Form payload.

    Returns:
        list[dict]: Each dict has 'factor' (str) and 'severity' (str).
                    severity ∈ {'low', 'moderate', 'high', 'critical'}
    """
    factors = []

    # --- Age ---
    age = int(data.get('age', 0))
    if   age >= 70: factors.append({'factor': 'Age over 70', 'severity': 'critical'})
    elif age >= 60: factors.append({'factor': 'Age over 60', 'severity': 'high'})
    elif age >= 50: factors.append({'factor': 'Age over 50', 'severity': 'moderate'})

    # --- Blood Pressure (JNC-8 stages) ---
    sys = int(data.get('systolic', 0))
    dia = int(data.get('diastolic', 0))
    if   sys >= 180 or dia >= 120:
        factors.append({'factor': f'Hypertensive Crisis ({sys}/{dia} mmHg)', 'severity': 'critical'})
    elif sys >= 140 or dia >= 90:
        factors.append({'factor': f'Stage 2 Hypertension ({sys}/{dia} mmHg)', 'severity': 'high'})
    elif sys >= 130 or dia >= 80:
        factors.append({'factor': f'Stage 1 Hypertension ({sys}/{dia} mmHg)', 'severity': 'moderate'})
    elif sys >= 120 and dia < 80:
        factors.append({'factor': f'Elevated Blood Pressure ({sys}/{dia} mmHg)', 'severity': 'low'})

    # --- Cholesterol (NCEP ATP III) ---
    chol = int(data.get('cholesterol', 0))
    if   chol >= 240:
        factors.append({'factor': f'High Cholesterol ({chol} mg/dL)', 'severity': 'high'})
    elif chol >= 200:
        factors.append({'factor': f'Borderline High Cholesterol ({chol} mg/dL)', 'severity': 'moderate'})

    # --- Blood Sugar (ADA criteria) ---
    bs = int(data.get('bloodSugar', 0))
    if   bs >= 126:
        factors.append({'factor': f'Diabetic Range Blood Sugar ({bs} mg/dL)', 'severity': 'high'})
    elif bs >= 100:
        factors.append({'factor': f'Pre-diabetic Blood Sugar ({bs} mg/dL)', 'severity': 'moderate'})

    # --- BMI (WHO classification) ---
    bmi = float(data.get('bmi', 0))
    if   bmi >= 35:
        factors.append({'factor': f'Severely Obese BMI ({bmi} kg/m²)', 'severity': 'high'})
    elif bmi >= 30:
        factors.append({'factor': f'Obese BMI ({bmi} kg/m²)', 'severity': 'moderate'})
    elif bmi >= 25:
        factors.append({'factor': f'Overweight BMI ({bmi} kg/m²)', 'severity': 'low'})
    elif bmi < 18.5:
        factors.append({'factor': f'Underweight BMI ({bmi} kg/m²)', 'severity': 'low'})

    # --- Diabetes ---
    diabetes = data.get('diabetes', 'no')
    if   diabetes == 'yes':         factors.append({'factor': 'Diabetes Mellitus',  'severity': 'high'})
    elif diabetes == 'prediabetes': factors.append({'factor': 'Pre-diabetes',        'severity': 'moderate'})

    # --- Smoking ---
    smoking = data.get('smoking', 'never')
    if   smoking == 'current': factors.append({'factor': 'Current Smoker',  'severity': 'high'})
    elif smoking == 'former':  factors.append({'factor': 'Former Smoker',   'severity': 'low'})

    # --- Family History ---
    if data.get('familyHistory') == 'yes':
        factors.append({'factor': 'Family History of Heart Disease', 'severity': 'moderate'})

    # --- Previous Heart Issues ---
    if data.get('previousHeart') == 'yes':
        factors.append({'factor': 'Previous Heart Issues (Major Risk Factor)', 'severity': 'critical'})

    # --- Physical Activity ---
    activity = data.get('activity', 'moderate')
    if   activity == 'sedentary': factors.append({'factor': 'Sedentary Lifestyle',   'severity': 'moderate'})
    elif activity == 'light':     factors.append({'factor': 'Low Physical Activity',  'severity': 'low'})

    return factors


def probability_to_risk_level(probability: float) -> tuple:
    """
    Map predicted probability [0, 1] to risk level label and CSS class.

    Args:
        probability (float): Model output probability for class 1.

    Returns:
        tuple: (risk_level: str, level_class: str)
    """
    pct = probability * 100
    if   pct <= 25: return 'Low Risk',      'risk-low'
    elif pct <= 50: return 'Moderate Risk', 'risk-moderate'
    elif pct <= 75: return 'High Risk',     'risk-high'
    else:           return 'Critical Risk', 'risk-critical'


# ============================================================
# ROUTES — Static File Serving
# Flask serves all HTML, CSS, and JS files from the project folder
# ============================================================

@app.route('/')
def serve_root():
    """Root URL redirects to login page."""
    return send_from_directory('.', 'login.html')

@app.route('/index.html')
def serve_index():
    """Serve the main prediction dashboard."""
    return send_from_directory('.', 'index.html')

@app.route('/change-password.html')
def serve_change_password():
    """Serve the change-password settings page."""
    return send_from_directory('.', 'change-password.html')

@app.route('/login.html')
def serve_login():
    """Serve the login page explicitly."""
    return send_from_directory('.', 'login.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve any static file (style.css, script.js, etc.)."""
    return send_from_directory('.', filename)


# ============================================================
# API ROUTE: POST /predict
# Main prediction endpoint — called by script.js on form submit
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Logistic Regression Prediction Endpoint

    Accepts patient health data as JSON, runs it through the
    trained Logistic Regression model, and returns the
    heart failure risk prediction with confidence score.

    Request Body (JSON):
    ─────────────────────────────────────────────────────────
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

    Response (JSON):
    ─────────────────────────────────────────────────────────
    {
      "success": true,
      "risk_percentage": 71.3,
      "risk_level": "High Risk",
      "level_class": "risk-high",
      "confidence": 42.6,
      "probability": 0.713,
      "prediction": 1,
      "risk_factors": [
        {"factor": "Age over 50", "severity": "moderate"},
        ...
      ],
      "model_accuracy": 87.33,
      "auc_score": 93.10,
      "algorithm": "Logistic Regression"
    }
    """
    # --- Guard: ensure model is loaded ---
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'ML model not loaded. Run: python train_model.py'
        }), 503

    # --- Parse request body ---
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body received'}), 400

    # --- Validate input ---
    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({'success': False, 'error': error_msg}), 422

    try:
        # 1. Encode form strings → numeric feature vector
        features = encode_input(data)                          # shape (1, 12)

        # 2. Scale using fitted StandardScaler
        features_scaled = scaler.transform(features)           # shape (1, 12)

        # 3. Logistic Regression prediction
        prediction        = int(model.predict(features_scaled)[0])
        pred_probabilities = model.predict_proba(features_scaled)[0]
        probability       = float(pred_probabilities[1])       # P(heart failure)

        # 4. Derive display values
        risk_percentage = round(probability * 100, 1)
        risk_level, level_class = probability_to_risk_level(probability)

        # Confidence: distance from decision boundary (0.5)
        # 0% = model is uncertain, 100% = model is very confident
        confidence = round(abs(probability - 0.5) * 200, 1)

        # 5. Extract interpretable risk factors
        risk_factors = extract_risk_factors(data)

        # 6. Build response
        return jsonify({
            'success':        True,
            'risk_percentage': risk_percentage,
            'risk_level':     risk_level,
            'level_class':    level_class,
            'confidence':     confidence,
            'probability':    round(probability, 4),
            'prediction':     prediction,
            'risk_factors':   risk_factors,
            'model_accuracy': round(model_info.get('accuracy', 0) * 100, 2),
            'auc_score':      round(model_info.get('auc_score', 0) * 100, 2),
            'algorithm':      'Logistic Regression'
        }), 200

    except Exception as exc:
        print(f"  ❌ Prediction error: {exc}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(exc)}'}), 500


# ============================================================
# API ROUTE: GET /model-info
# Returns model performance metrics for display in the frontend
# ============================================================

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Returns model accuracy, AUC-ROC, confusion matrix, and
    feature importance coefficients. Called on page load by script.js.
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Run train_model.py.'}), 503

    return jsonify({
        'algorithm':          model_info.get('algorithm', 'Logistic Regression'),
        'framework':          model_info.get('framework', 'scikit-learn'),
        'accuracy':           model_info.get('accuracy', 0),
        'accuracy_pct':       round(model_info.get('accuracy', 0) * 100, 2),
        'precision':          model_info.get('precision', 0),
        'recall':             model_info.get('recall', 0),
        'f1_score':           model_info.get('f1_score', 0),
        'auc_score':          model_info.get('auc_score', 0),
        'auc_pct':            round(model_info.get('auc_score', 0) * 100, 2),
        'cv_mean':            model_info.get('cv_mean', 0),
        'cv_std':             model_info.get('cv_std', 0),
        'confusion_matrix':   model_info.get('confusion_matrix', []),
        'n_training_samples': model_info.get('n_training_samples', 0),
        'n_test_samples':     model_info.get('n_test_samples', 0),
        'feature_importance': model_info.get('feature_importance', [])
    }), 200


# ============================================================
# API ROUTE: GET /health
# Simple health-check — confirms server and model are running
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status':       'running',
        'model_loaded': MODEL_LOADED,
        'algorithm':    'Logistic Regression',
        'version':      '2.0.0'
    }), 200


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  CardioPredict — Flask API Server  v2.0")
    print("=" * 55)
    print(f"  Model Status   : {'✅ Loaded' if MODEL_LOADED else '❌ Not loaded (run train_model.py)'}")

    if MODEL_LOADED:
        acc = model_info.get('accuracy', 0) * 100
        auc = model_info.get('auc_score', 0) * 100
        print(f"  Model Accuracy : {acc:.2f}%")
        print(f"  AUC-ROC Score  : {auc:.2f}%")
        print(f"  Training Data  : {model_info.get('n_training_samples', 0)} samples")

    print("\n  Server URL     : http://localhost:5000")
    print("  Default Login  : admin / admin123")
    print("=" * 55 + "\n")

    app.run(
        host='0.0.0.0',   # Accept connections from any network interface
        port=5000,
        debug=True         # Auto-reload on code changes; disable for production
    )
