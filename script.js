// ===================================================================
//  CardioPredict — Frontend Logic  (v2.0)
//  Machine Learning Edition — Logistic Regression via Flask API
// ===================================================================
//
//  What changed from v1.0:
//  - calculateRisk() replaced with async fetchPrediction() API call
//  - Flask backend (app.py) now handles prediction via Logistic Regression
//  - displayResults() updated to show confidence score
//  - loadModelInfo() fetches model accuracy on page load
//  - showLoading / showAPIError added for better UX
//
//  Architecture:
//  Browser Form → POST /predict (Flask) → Logistic Regression → JSON → UI
// ===================================================================

// -------------------------------------------------------------------
// CONFIGURATION
// -------------------------------------------------------------------

/** Flask API base URL. The app is served BY Flask at port 5000. */
const API_BASE_URL = 'http://localhost:5000';

// -------------------------------------------------------------------
// DOM ELEMENT REFERENCES
// -------------------------------------------------------------------

const healthForm      = document.getElementById('healthForm');
const resultsSection  = document.getElementById('resultsSection');
const riskIndicator   = document.getElementById('riskIndicator');
const riskIcon        = document.getElementById('riskIcon');
const riskLevel       = document.getElementById('riskLevel');
const riskPercentage  = document.getElementById('riskPercentage');
const factorsList     = document.getElementById('factorsList');
const adviceList      = document.getElementById('adviceList');
const emergencyAlert  = document.getElementById('emergencyAlert');


// ===================================================================
// ON PAGE LOAD: Fetch model performance stats from Flask
// Populates the "Model Performance" info card in the right panel.
// ===================================================================

document.addEventListener('DOMContentLoaded', async function () {
    await loadModelInfo();
});

/**
 * Fetches model accuracy, AUC, and training info from GET /model-info
 * and updates the Model Performance card in the right info panel.
 */
async function loadModelInfo() {
    const elAccuracy  = document.getElementById('mlAccuracy');
    const elAuc       = document.getElementById('mlAucScore');
    const elSamples   = document.getElementById('mlTrainSamples');
    const elStatus    = document.getElementById('mlServerStatus');

    try {
        const response = await fetch(`${API_BASE_URL}/model-info`, {
            method: 'GET',
            signal: AbortSignal.timeout(4000)   // 4-second timeout
        });

        if (!response.ok) throw new Error('Server responded with error');

        const info = await response.json();

        // Update model performance card
        if (elAccuracy) elAccuracy.textContent  = `${info.accuracy_pct}%`;
        if (elAuc)      elAuc.textContent        = `${info.auc_pct}%`;
        if (elSamples)  elSamples.textContent    = info.n_training_samples.toLocaleString();
        if (elStatus) {
            elStatus.textContent  = '● Online';
            elStatus.style.color  = '#10b981';
        }

    } catch (err) {
        // Server is offline — show graceful degraded state
        if (elAccuracy)  elAccuracy.textContent  = '—';
        if (elAuc)       elAuc.textContent        = '—';
        if (elSamples)   elSamples.textContent    = '—';
        if (elStatus) {
            elStatus.textContent = '● Offline  (run: python app.py)';
            elStatus.style.color = '#ef4444';
        }
        console.warn('CardioPredict: Flask server not reachable.', err.message);
    }
}


// ===================================================================
// FORM SUBMIT — ASYNC ML PREDICTION
// Replaces the old synchronous calculateRisk() call.
// Flow: collect form data → POST /predict → displayResults()
// ===================================================================

healthForm.addEventListener('submit', async function (e) {
    e.preventDefault();

    // 1. Collect all form field values
    const formData = {
        age:           parseInt(document.getElementById('age').value),
        gender:        document.getElementById('gender').value,
        systolic:      parseInt(document.getElementById('systolic').value),
        diastolic:     parseInt(document.getElementById('diastolic').value),
        cholesterol:   parseInt(document.getElementById('cholesterol').value),
        bloodSugar:    parseInt(document.getElementById('bloodSugar').value),
        bmi:           parseFloat(document.getElementById('bmi').value),
        activity:      document.getElementById('activity').value,
        diabetes:      document.querySelector('input[name="diabetes"]:checked').value,
        smoking:       document.querySelector('input[name="smoking"]:checked').value,
        familyHistory: document.querySelector('input[name="familyHistory"]:checked').value,
        previousHeart: document.querySelector('input[name="previousHeart"]:checked').value
    };

    // 2. Enter loading state (disable button, show spinner)
    setLoadingState(true);

    try {
        // 3. Call Flask API — POST /predict
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(formData),
            signal:  AbortSignal.timeout(10000)   // 10-second timeout
        });

        const result = await response.json();

        // 4. Handle API error response
        if (!result.success) {
            throw new Error(result.error || 'Prediction returned an error.');
        }

        // 5. Map API response to the shape displayResults() expects
        const riskResult = {
            percentage:    result.risk_percentage,  // 0–100 (from probability × 100)
            level:         result.risk_level,        // 'Low Risk' / 'Moderate Risk' / etc.
            levelClass:    result.level_class,        // CSS class name
            factors:       result.risk_factors,       // array of {factor, severity}
            confidence:    result.confidence,         // 0–100% model confidence
            modelAccuracy: result.model_accuracy,     // e.g. 87.33
            aucScore:      result.auc_score           // e.g. 93.10
        };

        // 6. Render results in the existing UI
        displayResults(riskResult, formData);

        // 7. Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        console.error('CardioPredict API error:', err);

        // Show a friendly error in the results panel
        const isOffline = err.message.toLowerCase().includes('fetch') ||
                          err.name === 'TimeoutError' ||
                          err.name === 'AbortError';

        const msg = isOffline
            ? '⚠️ Cannot reach Flask server.\nPlease run: python app.py'
            : `⚠️ ${err.message}`;

        showAPIError(msg);

    } finally {
        // 8. Always restore button state
        setLoadingState(false);
    }
});


// ===================================================================
// DISPLAY RESULTS
// Renders the ML prediction result in the existing results panel.
// Signature unchanged from v1.0 so all existing CSS/HTML still works.
// ===================================================================

/**
 * Populate the results panel with the ML prediction output.
 *
 * @param {Object} result - From the /predict API response
 * @param {Object} formData - The collected form values
 */
function displayResults(result, formData) {
    // Show the results section (was hidden)
    resultsSection.classList.add('show');

    // --- Risk indicator panel ---
    riskIndicator.className = `risk-indicator ${result.levelClass}`;

    const icons = {
        'risk-low':      '💚',
        'risk-moderate': '💛',
        'risk-high':     '🧡',
        'risk-critical': '❤️'
    };
    riskIcon.textContent       = icons[result.levelClass] || '❤️';
    riskLevel.textContent      = result.level;
    riskPercentage.textContent = `ML Risk Score: ${result.percentage}%`;

    // --- Confidence + model accuracy line ---
    const confidenceEl = document.getElementById('confidenceDisplay');
    if (confidenceEl) {
        const confText = document.getElementById('confidenceText');
        if (confText) {
            confText.textContent =
                `Model Confidence: ${result.confidence}%   |   ` +
                `Logistic Regression Accuracy: ${result.modelAccuracy}%   |   ` +
                `AUC: ${result.aucScore}%`;
        }
        confidenceEl.style.display = 'block';
    }

    // --- Risk factors list ---
    factorsList.innerHTML = '';

    if (result.factors && result.factors.length > 0) {
        result.factors.forEach(function (factor) {
            const item = document.createElement('div');
            item.className = 'factor-item';
            item.innerHTML = `
                <span class="icon">${getSeverityIcon(factor.severity)}</span>
                <span>${factor.factor}</span>
            `;
            factorsList.appendChild(item);
        });
    } else {
        factorsList.innerHTML =
            '<p style="color: var(--success-color);">' +
            '✅ No significant risk factors identified. Keep up the healthy lifestyle!' +
            '</p>';
    }

    // --- Medical advice (generated client-side from form data) ---
    const advice = generateAdvice(result, formData);
    adviceList.innerHTML = '';
    advice.forEach(function (item) {
        const adviceItem = document.createElement('div');
        adviceItem.className = 'advice-item';
        adviceItem.innerHTML = `
            <h4>${item.icon} ${item.title}</h4>
            <ul>
                ${item.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        adviceList.appendChild(adviceItem);
    });

    // --- Emergency alert for critical/high risk ---
    emergencyAlert.style.display =
        (result.levelClass === 'risk-critical' || result.percentage >= 70)
            ? 'block'
            : 'none';
}


// ===================================================================
// UI HELPERS
// ===================================================================

/**
 * Toggle the submit button between normal and loading state.
 * @param {boolean} isLoading
 */
function setLoadingState(isLoading) {
    const btn = document.querySelector('.btn-primary');
    if (!btn) return;

    if (isLoading) {
        btn.disabled   = true;
        btn.innerHTML  = '<span>⏳</span> Analyzing with AI...';
        btn.style.opacity = '0.75';
    } else {
        btn.disabled   = false;
        btn.innerHTML  = '<span>🔍</span> Analyze My Heart Health';
        btn.style.opacity = '1';
    }
}

/**
 * Display a Flask connection / server error in the results panel.
 * @param {string} message - Error message to show
 */
function showAPIError(message) {
    resultsSection.classList.add('show');
    riskIndicator.className    = 'risk-indicator risk-moderate';
    riskIcon.textContent       = '⚠️';
    riskLevel.textContent      = 'Server Error';
    riskPercentage.textContent = message;
    factorsList.innerHTML      =
        '<p style="color:var(--warning-color);">' +
        'Please make sure the Flask server is running:<br>' +
        '<code style="background:rgba(0,0,0,0.3);padding:4px 8px;border-radius:4px;">' +
        'python app.py</code></p>';
    adviceList.innerHTML = '';
    emergencyAlert.style.display = 'none';

    const confidenceEl = document.getElementById('confidenceDisplay');
    if (confidenceEl) confidenceEl.style.display = 'none';
}


// ===================================================================
// HELPER: Severity → Emoji Icon
// ===================================================================

/**
 * Maps a risk severity string to an emoji icon.
 * @param {string} severity - 'low' | 'moderate' | 'high' | 'critical'
 * @returns {string} Emoji
 */
function getSeverityIcon(severity) {
    const icons = {
        'low':      '⚠️',
        'moderate': '⚠️',
        'high':     '🔴',
        'critical': '🚨'
    };
    return icons[severity] || '⚠️';
}


// ===================================================================
// MEDICAL ADVICE GENERATOR (client-side — unchanged from v1.0)
// Generates personalised recommendations from form data.
// ===================================================================

/**
 * Generate tailored medical advice categories based on patient inputs.
 *
 * @param {Object} result   - Risk result from API
 * @param {Object} formData - Raw form values
 * @returns {Array<Object>} advice items with icon, title, recommendations
 */
function generateAdvice(result, formData) {
    const advice = [];

    // --- Lifestyle Modifications ---
    const lifestyleRecs = [];
    if (formData.activity === 'sedentary' || formData.activity === 'light') {
        lifestyleRecs.push('Increase physical activity to at least 150 minutes of moderate exercise per week');
        lifestyleRecs.push('Start with simple activities like walking, swimming, or cycling');
    }
    if (formData.smoking === 'current') {
        lifestyleRecs.push('Quit smoking immediately — this is the single most impactful step you can take');
        lifestyleRecs.push('Consider nicotine replacement therapy or a smoking cessation programme');
    }
    if (formData.bmi >= 25) {
        lifestyleRecs.push('Work towards a healthy weight through balanced diet and regular exercise');
        lifestyleRecs.push('Aim for gradual weight loss of 0.5–1 kg per week');
    }
    if (lifestyleRecs.length > 0) {
        advice.push({ icon: '🏃‍♂️', title: 'Lifestyle Modifications', recommendations: lifestyleRecs });
    }

    // --- Dietary Recommendations ---
    const dietRecs = [];
    if (formData.cholesterol >= 200) {
        dietRecs.push('Reduce saturated fats and eliminate trans fats from your diet');
        dietRecs.push('Increase omega-3 fatty acids: oily fish, walnuts, flaxseeds');
    }
    if (formData.bloodSugar >= 100 || formData.diabetes !== 'no') {
        dietRecs.push('Monitor carbohydrate intake; prefer complex carbs over refined sugars');
        dietRecs.push('Eat regular balanced meals to keep blood sugar stable');
    }
    if (formData.systolic >= 130 || formData.diastolic >= 80) {
        dietRecs.push('Reduce sodium intake to under 2,300 mg per day');
        dietRecs.push('Follow the DASH diet (Dietary Approaches to Stop Hypertension)');
    }
    dietRecs.push('Eat plenty of fruits, vegetables, whole grains, and lean proteins');
    dietRecs.push('Limit alcohol consumption to moderate levels');
    advice.push({ icon: '🥗', title: 'Dietary Recommendations', recommendations: dietRecs });

    // --- Medical Follow-up ---
    const medRecs = [];
    if (result.percentage >= 50) {
        medRecs.push('Schedule an appointment with a cardiologist for comprehensive evaluation');
        medRecs.push('Discuss medication options (statins, antihypertensives) with your doctor');
    } else if (result.percentage >= 25) {
        medRecs.push('Consult your primary care physician about your identified risk factors');
        medRecs.push('Schedule regular cardiovascular check-ups every 6–12 months');
    } else {
        medRecs.push('Continue annual health screenings');
        medRecs.push('Maintain open communication with your healthcare provider');
    }
    medRecs.push('Monitor your blood pressure at home regularly');
    medRecs.push('Keep a log of your health metrics and share with your doctor');
    advice.push({ icon: '👨‍⚕️', title: 'Medical Follow-up', recommendations: medRecs });

    // --- Stress Management ---
    advice.push({
        icon: '🧘‍♀️',
        title: 'Stress Management',
        recommendations: [
            'Practice relaxation techniques: meditation, deep breathing, or yoga',
            'Ensure adequate sleep (7–9 hours per night)',
            'Engage in hobbies and social activities to reduce mental stress',
            'Consider counselling or support groups if stress is severe'
        ]
    });

    // --- Monitoring & Prevention ---
    const monRecs = [
        'Keep a personal health journal to track progress',
        'Set realistic and measurable health goals',
        'Stay informed about cardiovascular health and prevention strategies'
    ];
    if (formData.diabetes !== 'no') {
        monRecs.push('Monitor blood glucose levels daily as recommended by your doctor');
    }
    if (formData.previousHeart === 'yes') {
        monRecs.push('Follow your cardiac rehabilitation programme diligently');
        monRecs.push('Take all prescribed medications exactly as directed — do not skip doses');
    }
    advice.push({ icon: '📊', title: 'Monitoring & Prevention', recommendations: monRecs });

    return advice;
}


// ===================================================================
// RESET FORM
// ===================================================================

/** Clear the form and hide the results section. */
function resetForm() {
    healthForm.reset();
    resultsSection.classList.remove('show');

    // Hide confidence display
    const confidenceEl = document.getElementById('confidenceDisplay');
    if (confidenceEl) confidenceEl.style.display = 'none';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}


// ===================================================================
// SMOOTH SCROLL for anchor links
// ===================================================================

document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
});


// ===================================================================
// FORM VALIDATION — real-time border colour feedback
// ===================================================================

document.querySelectorAll('input[type="number"], select').forEach(function (input) {
    input.addEventListener('blur', function () {
        if (this.value && this.checkValidity()) {
            this.style.borderColor = 'var(--success-color)';
        } else if (this.value) {
            this.style.borderColor = 'var(--danger-color)';
        }
    });

    input.addEventListener('focus', function () {
        this.style.borderColor = 'var(--primary-color)';
    });
});


// ===================================================================
// CONSOLE BRANDING
// ===================================================================

console.log('%c❤️  CardioPredict v2.0 — ML Edition', 'color:#3b82f6;font-size:20px;font-weight:bold;');
console.log('%c🤖 Powered by Logistic Regression (scikit-learn)', 'color:#10b981;font-size:14px;');
console.log('%c🌐 Flask API: ' + API_BASE_URL, 'color:#f59e0b;font-size:12px;');
console.log('%c⚠️  For educational purposes only. Consult healthcare professionals.', 'color:#94a3b8;font-size:11px;');
