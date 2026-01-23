import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
MODEL_PATH = "Auto_Viz_marketing_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Auto-Viz Marketing Model API is running."

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

        # ---- Create DataFrame ----
        input_df = pd.DataFrame([{
            "TV": float(tv),
            "Radio": float(radio),
            "Newspaper": float(newspaper)
        }])

        # ---- Prediction ----
        prediction = model.predict(input_df)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

