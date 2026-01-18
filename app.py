from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# [cite_start]Load the model 
MODEL_PATH = 'Auto_Viz_marketing_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

# [cite_start]The 12 features required by your model [cite: 1]
FEATURES = [
    'impressions', 'clicks', 'spend', 'hour', 'day_of_week', 
    'is_weekend', 'is_business_hours', 'impressions_lag_1', 
    'clicks_lag_1', 'conversions_lag_1', 
    'impressions_rolling_mean_24', 'clicks_rolling_mean_24'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if all required features are present
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400

        # Convert data to DataFrame for the model
        # Using a list of lists to ensure 2D array format
        input_data = pd.DataFrame([data])[FEATURES]

        # Generate prediction
        prediction = model.predict(input_data)

        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
