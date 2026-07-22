import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Absolute paths to ensure model & scaler load regardless of execution CWD
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form

    try:
        # Extract and parse feature inputs safely
        parsed_inputs = []
        for feature in FEATURE_NAMES:
            val = data.get(feature)
            if val is None or str(val).strip() == '':
                return jsonify({"error": f"Missing value for field: {feature}"}), 400
            parsed_inputs.append(float(val))

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid numeric input provided: {str(e)}"}), 400

    # Build input DataFrame matching training columns
    input_data = pd.DataFrame([parsed_inputs], columns=FEATURE_NAMES)

    # Scale inputs using fitted StandardScaler to prevent unscaled feature distortion
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=FEATURE_NAMES)

    # Predict target class and risk probability
    prediction = int(model.predict(input_scaled)[0])

    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(input_scaled)[0][1])
    else:
        prob = 1.0 if prediction == 1 else 0.0

    risk_percentage = round(prob * 100, 1)

    # Key clinical factors breakdown for UI
    risk_factors = []
    if input_data['trestbps'].iloc[0] > 140:
        risk_factors.append(f"Elevated Blood Pressure ({int(input_data['trestbps'].iloc[0])} mmHg)")
    if input_data['chol'].iloc[0] > 240:
        risk_factors.append(f"High Serum Cholesterol ({int(input_data['chol'].iloc[0])} mg/dL)")
    if input_data['thalach'].iloc[0] < 120:
        risk_factors.append(f"Reduced Max Heart Rate ({int(input_data['thalach'].iloc[0])} BPM)")
    if input_data['oldpeak'].iloc[0] >= 1.5:
        risk_factors.append(f"Significant ST Depression ({input_data['oldpeak'].iloc[0]})")
    if input_data['cp'].iloc[0] > 0:
        risk_factors.append("Anginal Chest Pain Present")

    response_payload = {
        "result": str(prediction),
        "probability": risk_percentage,
        "risk_level": "High" if prediction == 1 else "Low",
        "risk_factors": risk_factors
    }

    if request.is_json:
        return jsonify(response_payload)

    return render_template('index.html', result=str(prediction), probability=risk_percentage)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
