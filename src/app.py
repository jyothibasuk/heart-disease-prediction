import json
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = None
def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "cardio_model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully")

# Scoring endpoint
@app.route("/score", methods=["POST"])
def score():
    try:
        data = request.get_json(force=True)
        if "data" not in data:
            return jsonify({"error": "Missing 'data' key in JSON"}), 400
        
        input_data = np.array(data["data"])
        df = pd.DataFrame(input_data, columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Swagger JSON endpoint
@app.route("/swagger.json", methods=["GET"])
def swagger():
    swagger_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Cardio Model API",
            "description": "API for predicting heart disease risk",
            "version": "1.0.0"
        },
        "servers": [
            {"url": "/"}  # Base URL will be updated by Azure ML
        ],
        "paths": {
            "/score": {
                "post": {
                    "summary": "Predict heart disease risk",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "example": [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]
                                            }
                                        }
                                    },
                                    "required": ["data"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {"type": "array", "items": {"type": "integer"}}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            }
        }
    }
    return jsonify(swagger_spec)

if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5000)