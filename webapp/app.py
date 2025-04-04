# webapp/app.py
import json
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("cardio-model")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        result = "At Risk" if prediction == 1 else "No Risk"
        return json.dumps({"prediction": result})
    except Exception as e:
        return json.dumps({"error": str(e)})