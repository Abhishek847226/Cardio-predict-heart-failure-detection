@echo off
echo.
echo  ============================================================
echo   CardioPredict - One-Click Setup  (Windows)
echo  ============================================================
echo.

REM Step 1: Install Python dependencies
echo  [1/4] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo  ERROR: pip install failed. Make sure Python 3.8+ is installed.
    pause
    exit /b 1
)
echo  Done.

REM Step 2: Generate synthetic dataset
echo.
echo  [2/4] Generating heart failure dataset...
python generate_dataset.py
if %errorlevel% neq 0 (
    echo  ERROR: Dataset generation failed.
    pause
    exit /b 1
)

REM Step 3: Train the Logistic Regression model
echo.
echo  [3/4] Training Logistic Regression model...
python train_model.py
if %errorlevel% neq 0 (
    echo  ERROR: Model training failed.
    pause
    exit /b 1
)

REM Step 4: Start the Flask server
echo.
echo  [4/4] Starting Flask API server...
echo.
echo  ============================================================
echo   Open your browser and navigate to:  http://localhost:5000
echo   Default Login:  admin / admin123
echo  ============================================================
echo.
python app.py
pause
