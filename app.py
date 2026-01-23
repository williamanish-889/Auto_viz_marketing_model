import os
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
MODEL_PATH = "Auto_Viz_marketing_model.pkl"
model = joblib.load(MODEL_PATH)

# HTML template for browser testing
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Auto-Viz Marketing Model Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Marketing Sales Predictor</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label for="tv">TV Budget ($1000s):</label>
                <input type="number" step="0.01" id="tv" name="TV" placeholder="e.g., 230.1" required>
            </div>
            <div class="form-group">
                <label for="radio">Radio Budget ($1000s):</label>
                <input type="number" step="0.01" id="radio" name="Radio" placeholder="e.g., 37.8" required>
            </div>
            <div class="form-group">
                <label for="newspaper">Newspaper Budget ($1000s):</label>
                <input type="number" step="0.01" id="newspaper" name="Newspaper" placeholder="e.g., 69.2" required>
            </div>
            <button type="submit">Predict Sales</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                TV: parseFloat(document.getElementById('tv').value),
                Radio: parseFloat(document.getElementById('radio').value),
                Newspaper: parseFloat(document.getElementById('newspaper').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (response.ok) {
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `<strong>Predicted Sales:</strong> $${(data.prediction * 1000).toFixed(2)}`;
                } else {
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                }
                resultDiv.style.display = 'block';
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'error';
                resultDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # ---- GET: Browser query params ----
        if request.method == 'GET':
            tv = request.args.get("TV")
            radio = request.args.get("Radio")
            newspaper = request.args.get("Newspaper")
        # ---- POST: JSON body ----
        else:
            data = request.get_json(force=True)
            tv = data.get("TV")
            radio = data.get("Radio")
            newspaper = data.get("Newspaper")

        # ---- Validate inputs ----
        if tv is None or radio is None or newspaper is None:
            return jsonify({
                "error": "Missing input features. Required: TV, Radio, Newspaper"
            }), 400

        # ---- Convert to float and validate ----
        try:
            tv = float(tv)
            radio = float(radio)
            newspaper = float(newspaper)
        except ValueError:
            return jsonify({
                "error": "All inputs must be valid numbers"
            }), 400

        # ---- Create DataFrame ----
        input_df = pd.DataFrame([{
            "TV": tv,
            "Radio": radio,
            "Newspaper": newspaper
        }])

        # ---- Prediction ----
        prediction = model.predict(input_df)
        
        return jsonify({
            "prediction": float(prediction[0]),
            "inputs": {
                "TV": tv,
                "Radio": radio,
                "Newspaper": newspaper
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
