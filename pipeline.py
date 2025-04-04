import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from azureml.core import Workspace, Model, Environment, Experiment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Configuration
DATA_PATH = "data/heart.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/cardio_model.pkl"
WORKSPACE_CONFIG = "config.json"  # Download from Azure ML portal

def preprocess_data():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print("Data split into train and test sets.")

def train_model():
    df = pd.read_csv(TRAIN_PATH)
    X_train = df.drop("target", axis=1)
    y_train = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")

def evaluate_model():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)
    X_test = df.drop("target", axis=1)
    y_test = df["target"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {accuracy:.2f}")
    return accuracy

def deploy_to_azure_ml():
    # Connect to Azure ML Workspace
    ws = Workspace.from_config(WORKSPACE_CONFIG)
    print("Connected to Azure ML workspace:", ws.name)

    # Register the model
    model = Model.register(workspace=ws, model_path=MODEL_PATH, model_name="cardio-model")
    print("Model registered:", model.name, model.version)

    # Define environment
    env = Environment("cardio-env")
    env.python.conda_dependencies.add_conda_package("python=3.9")
    env.python.conda_dependencies.add_pip_package("scikit-learn")
    env.python.conda_dependencies.add_pip_package("pandas")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.python.conda_dependencies.add_pip_package("azureml-core")
    env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    env.register(ws)
    print("Environment registered:", env.name)

    # Define inference config
    inference_config = InferenceConfig(
        entry_script="webapp/app.py",
        environment=env
    )

    # Check if endpoint exists
    endpoint_name = "cardio-endpoint"
    try:
        service = AciWebservice(ws, endpoint_name)
        print("Endpoint exists, updating...")
        service.update(models=[model], inference_config=inference_config)
    except:
        print("Endpoint does not exist, creating...")
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            auth_enabled=True
        )
        service = Model.deploy(
            workspace=ws,
            name=endpoint_name,
            models=[model],
            inference_config=inference_config,
            deployment_config=deployment_config
        )
        service.wait_for_deployment(show=True)
    
    print("Deployment state:", service.state)
    print("Scoring URI:", service.scoring_uri)
    print("Authentication key:", service.get_keys()[0])

def run_pipeline():
    preprocess_data()
    train_model()
    accuracy = evaluate_model()
    if accuracy > 0.8:  # Deploy only if accuracy is acceptable
        deploy_to_azure_ml()
    else:
        print("Model accuracy too low, skipping deployment.")

if __name__ == "__main__":
    run_pipeline()