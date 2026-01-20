import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
MODEL_PATH = 'Auto_Viz_marketing_model.pkl'
model = joblib.load(MODEL_PATH)

# 1. ADD THIS: A "Home" route to stop the 404 errors
@app.route('/')
def home():
    return "Marketing Model API is running! Use the /predict endpoint for results."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # ... (your existing prediction code) ...
    return jsonify({"prediction": 0.0}) # Example placeholder

# 2. UPDATE THIS: Dynamic port binding for Render
if __name__ == '__main__':
    # Get port from environment variable, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # Must bind to 0.0.0.0 to be visible to Render's network
    app.run(host='0.0.0.0', port=port)
