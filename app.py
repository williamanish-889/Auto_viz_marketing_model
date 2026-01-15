import pickle
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

MODEL_PATH = 'Auto_Viz_marketing_model.pkl'

# Load the trained model
with open(MODEL_PATH, 'rb') as file:
    loaded_model = pickle.load(file)

# Define a simple home route for testing
@app.route('/', methods=['GET'])
def home():
    return "ML Model API is running!"

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided in the request body'}), 400

        required_columns = ['impressions', 'clicks', 'spend', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'impressions_lag_1', 'clicks_lag_1', 'conversions_lag_1', 'impressions_rolling_mean_24', 'clicks_rolling_mean_24']

        if isinstance(data, dict):
            input_df = pd.DataFrame([data], columns=required_columns)
        elif isinstance(data, list):
            input_df = pd.DataFrame(data, columns=required_columns)
        else:
            return jsonify({'error': 'Invalid input data format. Expected a JSON object or a list of JSON objects.'}), 400

        predictions = loaded_model.predict(input_df)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
