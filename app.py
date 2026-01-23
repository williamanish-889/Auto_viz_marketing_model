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
    <title>Auto-Viz Marketing Conversion Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            margin-bottom: 25px;
            border-radius: 8px;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: bold;
            font-size: 14px;
        }
        .input-wrapper {
            position: relative;
        }
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .helper-text {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        #result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            display: none;
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .success {
            background-color: #d4edda;
            border: 2px solid #c3e6cb;
            color: #155724;
        }
        .success .prediction-value {
            font-size: 32px;
            font-weight: bold;
            color: #28a745;
            text-align: center;
            margin: 15px 0;
        }
        .error {
            background-color: #f8d7da;
            border: 2px solid #f5c6cb;
            color: #721c24;
        }
        .result-details {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(0,0,0,0.1);
            font-size: 14px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        .metric-label {
            font-weight: bold;
        }
        .ctr-display {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Marketing Conversion Predictor</h1>
        <div class="subtitle">Predict conversions based on impressions and clicks</div>
        
        <div class="info">
            <strong>📊 Model Prediction:</strong> This model predicts the number of <strong>conversions</strong> you can expect based on your marketing campaign's impressions and clicks.
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="impressions">👁️ Impressions:</label>
                <div class="input-wrapper">
                    <input type="number" step="1" id="impressions" name="impressions" 
                           placeholder="e.g., 50000" required min="0">
                </div>
                <div class="helper-text">Total number of times your ad was displayed</div>
            </div>
            
            <div class="form-group">
                <label for="clicks">🖱️ Clicks:</label>
                <div class="input-wrapper">
                    <input type="number" step="1" id="clicks" name="clicks" 
                           placeholder="e.g., 1500" required min="0">
                </div>
                <div class="helper-text">Number of times users clicked on your ad</div>
            </div>
            
            <div class="ctr-display" id="ctrDisplay" style="display: none;">
                <strong>Click-Through Rate (CTR):</strong> <span id="ctrValue">0%</span>
            </div>
            
            <button type="submit">🚀 Predict Conversions</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        // Calculate and display CTR as user types
        function updateCTR() {
            const impressions = parseFloat(document.getElementById('impressions').value) || 0;
            const clicks = parseFloat(document.getElementById('clicks').value) || 0;
            
            if (impressions > 0 && clicks > 0) {
                const ctr = (clicks / impressions * 100).toFixed(2);
                document.getElementById('ctrValue').textContent = ctr + '%';
                document.getElementById('ctrDisplay').style.display = 'block';
            } else {
                document.getElementById('ctrDisplay').style.display = 'none';
            }
        }
        
        document.getElementById('impressions').addEventListener('input', updateCTR);
        document.getElementById('clicks').addEventListener('input', updateCTR);
        
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const impressions = parseFloat(document.getElementById('impressions').value);
            const clicks = parseFloat(document.getElementById('clicks').value);
            
            // Validation
            if (clicks > impressions) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'error';
                resultDiv.innerHTML = `<strong>⚠️ Validation Error:</strong> Clicks cannot exceed Impressions!`;
                resultDiv.style.display = 'block';
                return;
            }
            
            const formData = {
                impressions: impressions,
                clicks: clicks
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
                    const ctr = (clicks / impressions * 100).toFixed(2);
                    const conversionRate = (data.prediction / clicks * 100).toFixed(2);
                    
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `
                        <div style="text-align: center;">
                            <strong>✅ Predicted Conversions</strong>
                            <div class="prediction-value">${Math.round(data.prediction)}</div>
                        </div>
                        <div class="result-details">
                            <div class="metric">
                                <span class="metric-label">Impressions:</span>
                                <span>${impressions.toLocaleString()}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Clicks:</span>
                                <span>${clicks.toLocaleString()}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Click-Through Rate (CTR):</span>
                                <span>${ctr}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Conversion Rate:</span>
                                <span>${conversionRate}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Cost Per Conversion (if CPC=$1):</span>
                                <span>$${(clicks / data.prediction).toFixed(2)}</span>
                            </div>
                        </div>
                    `;
                } else {
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<strong>❌ Error:</strong> ${data.error}`;
                }
                resultDiv.style.display = 'block';
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'error';
                resultDiv.innerHTML = `<strong>❌ Error:</strong> ${error.message}`;
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
            impressions = request.args.get("impressions")
            clicks = request.args.get("clicks")
        # ---- POST: JSON body ----
        else:
            data = request.get_json(force=True)
            impressions = data.get("impressions")
            clicks = data.get("clicks")

        # ---- Validate inputs ----
        if impressions is None or clicks is None:
            return jsonify({
                "error": "Missing input features. Required: impressions, clicks"
            }), 400

        # ---- Convert to float and validate ----
        try:
            impressions = float(impressions)
            clicks = float(clicks)
        except ValueError:
            return jsonify({
                "error": "All inputs must be valid numbers"
            }), 400

        # Additional validation
        if impressions < 0 or clicks < 0:
            return jsonify({
                "error": "Impressions and clicks must be non-negative"
            }), 400

        if clicks > impressions:
            return jsonify({
                "error": "Clicks cannot exceed impressions"
            }), 400

        # ---- Create DataFrame with EXACT feature order from training ----
        # Your model was trained with x=df[["impressions", "clicks"]]
        # So we must maintain this exact order
        input_df = pd.DataFrame([[impressions, clicks]], 
                                columns=["impressions", "clicks"])

        # ---- Prediction ----
        prediction = model.predict(input_df)
        
        # Calculate additional metrics
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        
        return jsonify({
            "prediction": float(prediction[0]),
            "predicted_conversions": round(float(prediction[0])),
            "inputs": {
                "impressions": int(impressions),
                "clicks": int(clicks)
            },
            "metrics": {
                "ctr_percentage": round(ctr, 2),
                "conversion_rate_percentage": round((prediction[0] / clicks * 100), 2) if clicks > 0 else 0
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to check model feature names and details"""
    try:
        info = {
            "model_type": str(type(model).__name__),
            "prediction_target": "conversions"
        }
        
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            info["feature_names"] = list(model.feature_names_in_)
        else:
            info["expected_features"] = ["impressions", "clicks"]
        
        # Try to get model parameters
        if hasattr(model, 'get_params'):
            info["model_parameters"] = model.get_params()
        
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Auto-Viz Marketing Conversion Predictor API is running"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
