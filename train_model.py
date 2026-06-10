"""
============================================================
  CardioPredict - Logistic Regression Model Training Script
============================================================
This script trains, evaluates, and saves the Logistic Regression
model for heart failure risk prediction.

Pipeline:
  1. Load heart_failure_dataset.csv
  2. Exploratory Data Analysis
  3. Feature selection & StandardScaler normalization
  4. Train/Test split (80/20, stratified)
  5. Train Logistic Regression (scikit-learn)
  6. Evaluate: Accuracy, AUC-ROC, Confusion Matrix, CV Score
  7. Print Feature Coefficients (model interpretability)
  8. Save model.pkl, scaler.pkl, model_info.json

Run:  python train_model.py
============================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# --- scikit-learn imports ---
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# FEATURE COLUMN ORDER
# IMPORTANT: This order must match generate_dataset.py and app.py
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

TARGET_COLUMN = 'target_risk'


def print_section(title):
    """Helper to print a formatted section header."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print_section("STEP 1 — Loading Dataset")

try:
    df = pd.read_csv('cardiopredict heart failure detection.csv.csv')
    print(f"  ✅ Loaded {len(df)} records, {len(df.columns)} columns")
except FileNotFoundError:
    print("  ❌ 'cardiopredict heart failure detection.csv.csv' not found!")
    raise SystemExit(1)

# Drop missing values
before_len = len(df)
df = df.dropna()
missing_dropped = before_len - len(df)
if missing_dropped > 0:
    print(f"  ⚠️  Dropped {missing_dropped} rows with missing values.")

# Convert all feature columns and target column to numeric
for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any row that couldn't be converted to numeric
before_len = len(df)
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
conversion_dropped = before_len - len(df)
if conversion_dropped > 0:
    print(f"  ⚠️  Dropped {conversion_dropped} rows due to numeric conversion failures.")

# Clean outliers in blood pressure
before_len = len(df)
df = df[
    (df['systolic_bp'] >= 70) & (df['systolic_bp'] <= 250) &
    (df['diastolic_bp'] >= 40) & (df['diastolic_bp'] <= 150)
]
outliers_dropped = before_len - len(df)
if outliers_dropped > 0:
    print(f"  ⚠️  Dropped {outliers_dropped} rows with blood pressure outliers (systolic 70-250, diastolic 40-150).")


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print_section("STEP 2 — Exploratory Data Analysis")

print(f"  Shape          : {df.shape}")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"\n  Class Distribution (target = '{TARGET_COLUMN}'):")
counts = df[TARGET_COLUMN].value_counts()
for label, count in counts.items():
    name = "No Risk (0)" if label == 0 else "Heart Failure Risk (1)"
    print(f"    {name}: {count} records  ({count/len(df)*100:.1f}%)")

# Drop rows with missing values (safety check)
if df.isnull().any().any():
    before = len(df)
    df = df.dropna()
    print(f"\n  ⚠️  Dropped {before - len(df)} rows with missing values.")

print(f"\n  Feature Statistics:")
print(df[FEATURE_COLUMNS].describe().round(2).to_string())


# ============================================================
# STEP 3: PREPARE FEATURES AND TARGET
# ============================================================
print_section("STEP 3 — Feature Preparation")

X = df[FEATURE_COLUMNS]   # Input features  (shape: n x 12)
y = df[TARGET_COLUMN]     # Target variable  (shape: n x 1)

print(f"  Feature matrix X : {X.shape}  (samples × features)")
print(f"  Target vector  y : {y.shape}")
print(f"  Features used    : {FEATURE_COLUMNS}")


# ============================================================
# STEP 4: TRAIN / TEST SPLIT  (80% train, 20% test)
# Stratified split preserves the class balance in both sets.
# ============================================================
print_section("STEP 4 — Train/Test Split (80 / 20)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% held-out test set
    random_state=42,       # Reproducibility
    stratify=y             # Maintain class ratio in both splits
)

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Train class dist : {dict(y_train.value_counts().sort_index())}")
print(f"  Test  class dist : {dict(y_test.value_counts().sort_index())}")


# ============================================================
# STEP 5: FEATURE SCALING — StandardScaler
# Logistic Regression converges faster and more accurately
# when features are on the same scale: z = (x - mean) / std
# ============================================================
print_section("STEP 5 — Feature Scaling (StandardScaler)")

scaler = StandardScaler()

# Fit ONLY on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using training statistics (no re-fitting)
X_test_scaled  = scaler.transform(X_test)

print("  ✅ Features normalized: mean=0, std=1")
print(f"\n  Per-feature means (from training data):")
for feat, mean_val in zip(FEATURE_COLUMNS, scaler.mean_):
    print(f"    {feat:<28} {mean_val:.3f}")


# ============================================================
# STEP 6: TRAIN LOGISTIC REGRESSION
# Algorithm : Logistic Regression (binary classification)
# Loss      : Log-Loss / Binary Cross-Entropy
# Regularization : L2 (Ridge) — prevents overfitting
# Solver    : lbfgs (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
# ============================================================
print_section("STEP 6 — Training Logistic Regression")

print("  Algorithm        : Logistic Regression")
print("  Regularization   : L2 (Ridge),  C = 1.0")
print("  Solver           : lbfgs")
print("  Class weighting  : balanced (handles class imbalance)")
print("  Max iterations   : 1000")
print("\n  Training in progress...")

model = LogisticRegression(
    C=1.0,                  # Inverse regularization strength
    penalty='l2',           # L2 (Ridge) regularization
    solver='lbfgs',         # Efficient solver for L2
    max_iter=1000,          # Allow sufficient iterations
    random_state=42,        # Reproducibility
    class_weight='balanced' # Adjust for class imbalance
)

model.fit(X_train_scaled, y_train)
print("  ✅ Training complete!")


# ============================================================
# STEP 7: MODEL EVALUATION
# ============================================================
print_section("STEP 7 — Model Evaluation on Test Set")

# --- Predictions ---
y_pred      = model.predict(X_test_scaled)            # Binary: 0 or 1
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1

# --- Metrics ---
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc       = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)

# --- 5-Fold Stratified Cross-Validation ---
print("  Running 5-fold stratified cross-validation...")
X_all_scaled = scaler.transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_all_scaled, y, cv=cv, scoring='accuracy')

# --- Print Results ---
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │          MODEL PERFORMANCE REPORT        │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  Accuracy          :  {accuracy:.4f}  ({accuracy*100:.2f}%)   │")
print(f"  │  Precision         :  {precision:.4f}  ({precision*100:.2f}%)   │")
print(f"  │  Recall (Sensitivity): {recall:.4f}  ({recall*100:.2f}%)   │")
print(f"  │  F1-Score          :  {f1:.4f}              │")
print(f"  │  AUC-ROC Score     :  {auc:.4f}  ({auc*100:.2f}%)   │")
print(f"  │  CV Score (5-fold) :  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}   │")
print(f"  └─────────────────────────────────────────┘")

# Confusion Matrix
tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion Matrix:")
print(f"                    Predicted No    Predicted Yes")
print(f"  Actual No     :      {tn:>4}             {fp:>4}")
print(f"  Actual Yes    :      {fn:>4}             {tp:>4}")
print(f"\n  True Negatives  (TN): {tn}   (Correctly predicted No Risk)")
print(f"  False Positives (FP): {fp}   (No Risk but predicted Risk)")
print(f"  False Negatives (FN): {fn}   (Actual Risk but missed)")
print(f"  True Positives  (TP): {tp}   (Correctly predicted Risk)")

# Full Classification Report
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['No Risk', 'Heart Failure Risk'],
                             digits=4))


# ============================================================
# STEP 8: FEATURE IMPORTANCE (Logistic Regression Coefficients)
# A positive coefficient increases risk prediction;
# a negative coefficient decreases it.
# Magnitude indicates relative importance.
# ============================================================
print_section("STEP 8 — Feature Importance (Model Coefficients)")

coef_df = pd.DataFrame({
    'Feature':         FEATURE_COLUMNS,
    'Coefficient':     model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

print(f"\n  {'Rank':<5} {'Feature':<28} {'Coefficient':>12}  {'Effect'}")
print(f"  {'-'*60}")
for i, row in coef_df.iterrows():
    direction = "↑ Increases Risk" if row['Coefficient'] > 0 else "↓ Decreases Risk"
    print(f"  {i+1:<5} {row['Feature']:<28} {row['Coefficient']:>+12.4f}  {direction}")

print(f"\n  Model Intercept: {model.intercept_[0]:.4f}")


# ============================================================
# STEP 9: SAVE ARTIFACTS
# ============================================================
print_section("STEP 9 — Saving Model Artifacts")

# 1. Save trained Logistic Regression model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  ✅ model.pkl   — Trained Logistic Regression model")

# 2. Save fitted StandardScaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✅ scaler.pkl  — Fitted StandardScaler (mean & std from training data)")

# 3. Save model info as JSON (consumed by Flask API and frontend)
model_info = {
    "algorithm":           "Logistic Regression",
    "framework":           "scikit-learn",
    "accuracy":            float(accuracy),
    "precision":           float(precision),
    "recall":              float(recall),
    "f1_score":            float(f1),
    "auc_score":           float(auc),
    "cv_mean":             float(cv_scores.mean()),
    "cv_std":              float(cv_scores.std()),
    "confusion_matrix":    cm.tolist(),                  # [[TN, FP], [FN, TP]]
    "feature_names":       FEATURE_COLUMNS,
    "n_training_samples":  int(len(X_train)),
    "n_test_samples":      int(len(X_test)),
    "model_intercept":     float(model.intercept_[0]),
    "feature_importance":  coef_df[['Feature', 'Coefficient']].to_dict('records')
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("  ✅ model_info.json — Performance metrics & feature importance")

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*55}")
print(f"  ✅ Training complete!")
print(f"  Model Accuracy : {accuracy*100:.2f}%")
print(f"  AUC-ROC Score  : {auc*100:.2f}%")
print(f"\n  ▶  Next step: python app.py")
print(f"     Then open : http://localhost:5000")
print(f"{'='*55}\n")
