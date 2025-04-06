import json
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    global model
    try:
        model_path = Model.get_model_path("cardio-model")
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        # Log expected feature names for debugging
        print("Expected feature names:", model.feature_names_in_ if hasattr(model, 'feature_names_in_') else "Unknown")
    except Exception as e:
        print(f"Error in init(): {str(e)}")
        raise

def run(raw_data):
    try:
        print(f"Raw input received: {raw_data}")
        data = json.loads(raw_data)
        
        # Define expected feature names (based on training data)
        feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        
        # Handle different input formats
        if isinstance(data, dict):
            if "data" in data:  # Handle case where input is {"data": [...]}
                input_data = data["data"]
                if isinstance(input_data, list):
                    input_df = pd.DataFrame([input_data], columns=feature_names)
                else:
                    return json.dumps({"error": "Expected 'data' to be a list of values"})
            else:  # Handle direct dictionary input {"age": ..., "sex": ...}
                input_df = pd.DataFrame([data])
        elif isinstance(data, list):  # Handle direct list input [52, 1, 0, ...]
            input_df = pd.DataFrame([data], columns=feature_names)
        else:
            return json.dumps({"error": "Invalid input format"})
        
        # Verify feature names match
        missing_features = [f for f in feature_names if f not in input_df.columns]
        if missing_features:
            return json.dumps({"error": f"Missing features: {missing_features}"})
        
        print(f"Input DataFrame: {input_df.to_dict()}")  # Debug input
        prediction = model.predict(input_df)[0]
        result = "At Risk" if prediction == 1 else "No Risk"
        return json.dumps({"prediction": result})
    except Exception as e:
        return json.dumps({"error": str(e)})