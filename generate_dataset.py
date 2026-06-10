"""
============================================================
  CardioPredict - Heart Failure Dataset Generator
============================================================
Generates a synthetic-but-medically-realistic dataset of
1500 patient records for training the Logistic Regression
heart failure risk prediction model.

All feature distributions and risk weights are based on
published cardiovascular research literature.

Features Generated:
  age                 - Patient age (years)
  gender              - 1=Male, 0=Female
  systolic_bp         - Systolic blood pressure (mmHg)
  diastolic_bp        - Diastolic blood pressure (mmHg)
  cholesterol         - Total cholesterol (mg/dL)
  blood_sugar         - Fasting blood glucose (mg/dL)
  bmi                 - Body Mass Index (kg/m2)
  activity_level      - 0=Sedentary, 1=Light, 2=Moderate, 3=Active, 4=Very Active
  diabetes            - 0=No, 1=Pre-diabetes, 2=Yes
  smoking             - 0=Never, 1=Former, 2=Current
  family_history      - 0=No, 1=Unknown, 2=Yes
  previous_heart_issues - 0=No, 1=Yes

Target:
  heart_failure       - 0=Low Risk, 1=High Risk of Heart Failure

Output: heart_failure_dataset.csv
============================================================
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Reproducibility seed
np.random.seed(42)


def generate_heart_failure_dataset(n_samples=1500):
    """
    Generate a synthetic heart failure dataset with clinically
    realistic feature distributions and correlations.

    Args:
        n_samples (int): Number of patient records to generate.

    Returns:
        pd.DataFrame: Dataset with features and binary target column.
    """

    print(f"\n📋 Generating {n_samples} synthetic patient records...")

    # ----------------------------------------------------------------
    # STEP 1: SAMPLE EACH FEATURE FROM REALISTIC DISTRIBUTIONS
    # ----------------------------------------------------------------

    # Age: Normal distribution centered on 50 years
    age = np.random.normal(loc=50, scale=15, size=n_samples)
    age = np.clip(age, 18, 95).astype(int)

    # Gender: Slightly more males (heart disease is male-dominant)
    gender = np.random.binomial(1, 0.52, n_samples)          # 1=Male, 0=Female

    # Systolic Blood Pressure: Centered near elevated range
    systolic_bp = np.random.normal(loc=122, scale=18, size=n_samples)
    systolic_bp = np.clip(systolic_bp, 70, 250).astype(int)

    # Diastolic BP: Physiologically correlated with systolic
    diastolic_bp = 0.58 * systolic_bp + np.random.normal(15, 8, n_samples)
    diastolic_bp = np.clip(diastolic_bp, 40, 150).astype(int)

    # Total Cholesterol: mg/dL
    cholesterol = np.random.normal(loc=195, scale=42, size=n_samples)
    cholesterol = np.clip(cholesterol, 100, 400).astype(int)

    # Fasting Blood Sugar: mg/dL
    blood_sugar = np.random.normal(loc=90, scale=28, size=n_samples)
    blood_sugar = np.clip(blood_sugar, 50, 400).astype(int)

    # Body Mass Index
    bmi = np.random.normal(loc=25.5, scale=5.5, size=n_samples)
    bmi = np.clip(bmi, 10.0, 60.0).round(1)

    # Physical Activity Level (0=Sedentary to 4=Very Active)
    activity_probs = [0.28, 0.28, 0.25, 0.13, 0.06]
    activity_level = np.random.choice([0, 1, 2, 3, 4], n_samples, p=activity_probs)

    # Diabetes Status (0=No, 1=Pre-diabetes, 2=Diabetes)
    diabetes_probs = [0.68, 0.16, 0.16]
    diabetes = np.random.choice([0, 1, 2], n_samples, p=diabetes_probs)

    # Smoking Status (0=Never, 1=Former, 2=Current)
    smoking_probs = [0.58, 0.22, 0.20]
    smoking = np.random.choice([0, 1, 2], n_samples, p=smoking_probs)

    # Family History (0=No, 1=Unknown, 2=Yes)
    family_probs = [0.52, 0.16, 0.32]
    family_history = np.random.choice([0, 1, 2], n_samples, p=family_probs)

    # Previous Heart Issues
    previous_heart_issues = np.random.binomial(1, 0.10, n_samples)

    # ----------------------------------------------------------------
    # STEP 2: COMPUTE RISK SCORE (mirrors existing clinical weights)
    # This score is used ONLY to generate the binary target variable.
    # The Logistic Regression model will LEARN these relationships.
    # ----------------------------------------------------------------

    # Age contribution
    age_risk = np.where(age < 40, 0,
               np.where(age < 50, 5,
               np.where(age < 60, 10,
               np.where(age < 70, 15, 20))))

    # Gender contribution (males have higher baseline risk)
    gender_risk = gender * 5

    # Blood pressure contribution (JNC-8 hypertension stages)
    bp_risk = np.where((systolic_bp >= 180) | (diastolic_bp >= 120), 25,
              np.where((systolic_bp >= 140) | (diastolic_bp >= 90),  15,
              np.where((systolic_bp >= 130) | (diastolic_bp >= 80),  10,
              np.where((systolic_bp >= 120) & (diastolic_bp < 80),    5, 0))))

    # Cholesterol contribution (NCEP categories)
    chol_risk = np.where(cholesterol >= 240, 15,
                np.where(cholesterol >= 200,  8, 0))

    # Blood sugar contribution (ADA thresholds)
    bs_risk = np.where(blood_sugar >= 126, 15,
              np.where(blood_sugar >= 100,  8, 0))

    # BMI contribution (WHO obesity classification)
    bmi_risk = np.where(bmi >= 35, 18,
               np.where(bmi >= 30, 12,
               np.where(bmi >= 25,  5,
               np.where(bmi >= 18.5, 0, 3))))

    # Physical activity contribution (inverse: less activity = more risk)
    activity_risk = np.where(activity_level == 0, 15,
                   np.where(activity_level == 1, 10,
                   np.where(activity_level == 2,  5,
                   np.where(activity_level == 3,  2, 0))))

    # Diabetes contribution
    diabetes_risk = np.where(diabetes == 2, 20,
                   np.where(diabetes == 1, 10, 0))

    # Smoking contribution
    smoking_risk = np.where(smoking == 2, 15,
                  np.where(smoking == 1,  5, 0))

    # Family history contribution
    family_risk = np.where(family_history == 2, 10,
                 np.where(family_history == 1,  5, 0))

    # Previous heart issues contribution (strongest single factor)
    heart_risk = previous_heart_issues * 25

    # Sum all risk contributions
    total_risk = (age_risk + gender_risk + bp_risk + chol_risk + bs_risk +
                  bmi_risk + activity_risk + diabetes_risk + smoking_risk +
                  family_risk + heart_risk)

    # ----------------------------------------------------------------
    # STEP 3: ADD NOISE & CREATE BINARY TARGET
    # Gaussian noise simulates real-world unpredictability in health data.
    # Threshold of 45 gives ~42% positive class (heart failure risk).
    # ----------------------------------------------------------------

    noise = np.random.normal(0, 8, n_samples)
    total_risk_noisy = total_risk + noise
    heart_failure = (total_risk_noisy >= 45).astype(int)

    # ----------------------------------------------------------------
    # STEP 4: ASSEMBLE DATAFRAME
    # Column order MUST match what app.py and train_model.py expect.
    # ----------------------------------------------------------------

    df = pd.DataFrame({
        'age':                  age,
        'gender':               gender,
        'systolic_bp':          systolic_bp,
        'diastolic_bp':         diastolic_bp,
        'cholesterol':          cholesterol,
        'blood_sugar':          blood_sugar,
        'bmi':                  bmi,
        'activity_level':       activity_level,
        'diabetes':             diabetes,
        'smoking':              smoking,
        'family_history':       family_history,
        'previous_heart_issues': previous_heart_issues,
        'heart_failure':        heart_failure          # TARGET COLUMN
    })

    return df, heart_failure


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    # Generate the dataset
    df, target = generate_heart_failure_dataset(n_samples=1500)

    # ----------------------------------------------------------------
    # PRINT STATISTICS
    # ----------------------------------------------------------------
    n_positive = target.sum()
    n_negative = len(target) - n_positive

    print("\n📊 Dataset Statistics:")
    print(f"   Total Records       : {len(df)}")
    print(f"   Heart Failure (1)   : {n_positive} samples  ({n_positive/len(df)*100:.1f}%)")
    print(f"   No Risk (0)         : {n_negative} samples  ({n_negative/len(df)*100:.1f}%)")
    print(f"\n   Feature Summary:")
    print(f"   Age         : {df['age'].min()}–{df['age'].max()} yrs  (mean={df['age'].mean():.1f})")
    print(f"   Systolic BP : {df['systolic_bp'].min()}–{df['systolic_bp'].max()} mmHg (mean={df['systolic_bp'].mean():.1f})")
    print(f"   Cholesterol : {df['cholesterol'].min()}–{df['cholesterol'].max()} mg/dL (mean={df['cholesterol'].mean():.1f})")
    print(f"   BMI         : {df['bmi'].min()}–{df['bmi'].max()}  (mean={df['bmi'].mean():.1f})")

    # ----------------------------------------------------------------
    # SAVE CSV
    # ----------------------------------------------------------------
    csv_path = 'heart_failure_dataset.csv'
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Dataset saved: '{csv_path}'")
    print(f"   Columns : {list(df.columns)}")
    print("\n▶  Next step: python train_model.py\n")
