import os
import json
from azureml.core import Workspace, Model, Environment, Experiment, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Configuration
DATA_PATH = "../data/heart.csv"
TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"
MODEL_PATH = "../models/cardio_model.pkl"
WORKSPACE_CONFIG = "../config.json"  # Optional

def get_workspace():
    creds = os.environ.get("AZURE_CREDENTIALS")
    if creds:
        creds_dict = json.loads(creds)
        sp_auth = ServicePrincipalAuthentication(
            tenant_id=creds_dict["tenantId"],
            service_principal_id=creds_dict["clientId"],
            service_principal_password=creds_dict["clientSecret"]
        )
        ws = Workspace(
            subscription_id=creds_dict["subscriptionId"],
            resource_group="mlopsrg",
            workspace_name="risk-ml",
            auth=sp_auth
        )
        print("Authenticated using service principal from AZURE_CREDENTIALS.")
        return ws
    elif os.path.exists(WORKSPACE_CONFIG):
        ws = Workspace.from_config(WORKSPACE_CONFIG)
        print("Authenticated using config.json.")
        return ws
    else:
        raise Exception("No valid credentials found. Set AZURE_CREDENTIALS env var or provide config.json.")

def create_compute_target(ws):
    compute_name = "cpu-cluster"
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print("Found existing compute target:", compute_name)
    except ComputeTargetException:
        print("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_DS2_v2",
            min_nodes=0,
            max_nodes=4
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show=True)
    return compute_target

def build_pipeline(ws, compute_target):
    # Register dataset
    if "heart_data" not in ws.datasets:
        dataset = Dataset.File.from_files(path=DATA_PATH)
        dataset.register(ws, name="heart_data")
    dataset = ws.datasets["heart_data"]

    # Define environment
    env = Environment("cardio-env")
    env.python.conda_dependencies.add_conda_package("python=3.9")
    env.python.conda_dependencies.add_pip_package("scikit-learn")
    env.python.conda_dependencies.add_pip_package("pandas")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    env.register(ws)

    # Step 1: Preprocess Data
    preprocess_step = PythonScriptStep(
        name="Preprocess Data",
        script_name="preprocess.py",
        source_directory="src",
        compute_target=compute_target,
        arguments=["--data_path", dataset.as_mount(), "--train_path", TRAIN_PATH, "--test_path", TEST_PATH],
        runconfig=env.run_config,
        allow_reuse=True
    )

    # Step 2: Train Model
    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        source_directory="src",
        compute_target=compute_target,
        arguments=["--train_path", TRAIN_PATH, "--model_path", MODEL_PATH],
        runconfig=env.run_config,
        allow_reuse=False
    )

    # Step 3: Evaluate Model
    evaluate_step = PythonScriptStep(
        name="Evaluate Model",
        script_name="evaluate.py",
        source_directory="src",
        compute_target=compute_target,
        arguments=["--model_path", MODEL_PATH, "--test_path", TEST_PATH],
        runconfig=env.run_config,
        allow_reuse