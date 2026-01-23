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
        # ---- HANDLE GET (Browser) ----
        if request.method == 'GET':
            data = request.args
        else:
            data = request.get_json(force=True)

        input_df = pd.DataFrame([{
            "TV": float(data["TV"]),
            "Radio": float(data["Radio"]),
            "Newspaper": float(data["Newspaper"])
        }])

        prediction = model.predict(input_df)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except KeyError:
        return jsonify({
            "error": "Missing input features. Required: TV, Radio, Newspaper"
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

