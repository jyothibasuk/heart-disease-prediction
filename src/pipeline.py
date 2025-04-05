import os
import json
from azureml.core import Workspace, Model, Environment, Experiment, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Configuration
DATA_PATH = "data/heart.csv"  # Path in default datastore
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/cardio_model.pkl"

def get_workspace():
    # First try to authenticate using service principal from AZURE_CREDENTIALS
    creds = os.environ.get("AZURE_CREDENTIALS")
    if creds:
        try:
            creds_dict = json.loads(creds)
            sp_auth = ServicePrincipalAuthentication(
                tenant_id=creds_dict["tenantId"],
                service_principal_id=creds_dict["clientId"],
                service_principal_password=creds_dict["clientSecret"]
            )
            subscription_id = os.environ.get("AZUREML_SUBSCRIPTION_ID") or creds_dict["subscriptionId"]
            resource_group = os.environ.get("AZUREML_RESOURCE_GROUP") or "mlopsrg"
            workspace_name = os.environ.get("AZUREML_WORKSPACE_NAME") or "risk-ml"
            ws = Workspace(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name,
                auth=sp_auth
            )
            print(f"Authenticated using service principal with workspace: {workspace_name}")
            return ws
        except Exception as e:
            print(f"Failed to use AZURE_CREDENTIALS: {e}")

    # Try to authenticate using environment variables directly
    try:
        subscription_id = os.environ.get("AZUREML_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZUREML_RESOURCE_GROUP")
        workspace_name = os.environ.get("AZUREML_WORKSPACE_NAME")
        if subscription_id and resource_group and workspace_name:
            ws = Workspace(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name
            )
            print(f"Authenticated using environment variables with workspace: {workspace_name}")
            return ws
    except Exception as e:
        print(f"Failed to authenticate using environment variables: {e}")

    # Final fallback to config.json if it exists
    if os.path.exists("config.json"):
        try:
            ws = Workspace.from_config("config.json")
            print("Authenticated using config.json.")
            return ws
        except Exception as e:
            print(f"Failed to load config.json: {e}")
    
    raise Exception("No valid credentials found. Set AZURE_CREDENTIALS or AZUREML_* env variables.")

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
    # Get default datastore and upload data
    datastore = ws.get_default_datastore()
    if "heart_data" not in ws.datasets:
        datastore.upload_files(
            files=["data/heart.csv"],
            target_path="data/",
            overwrite=True,
            show_progress=True
        )
        dataset = Dataset.File.from_files(path=(datastore, DATA_PATH))
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

    # Define output for accuracy
    accuracy_output = PipelineData("accuracy_output", datastore=datastore)

    # Step 1: Preprocess Data
    preprocess_step = PythonScriptStep(
        name="Preprocess Data",
        script_name="preprocess.py",
        arguments=["--data_path", dataset.as_mount(), "--train_path", TRAIN_PATH, "--test_path", TEST_PATH],
        source_directory="src",
        compute_target=compute_target,
        runconfig=env.run_config,
        allow_reuse=True
    )

    # Step 2: Train Model
    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        arguments=["--train_path", TRAIN_PATH, "--model_path", MODEL_PATH],
        source_directory="src",
        compute_target=compute_target,
        runconfig=env.run_config,
        allow_reuse=False
    )

    # Step 3: Evaluate Model with output
    evaluate_step = PythonScriptStep(
        name="Evaluate Model",
        script_name="evaluate.py",
        arguments=["--model_path", MODEL_PATH, "--test_path", TEST_PATH, "--output", accuracy_output],
        outputs=[accuracy_output],
        source_directory="src",
        compute_target=compute_target,
        runconfig=env.run_config,
        allow_reuse=False
    )

    pipeline = Pipeline(workspace=ws, steps=[preprocess_step, train_step, evaluate_step])
    return pipeline, accuracy_output

def deploy_model(ws):
    model = Model.register(workspace=ws, model_path=MODEL_PATH, model_name="cardio-model")
    print("Model registered:", model.name, model.version)

    env = Environment.get(ws, "cardio-env")
    inference_config = InferenceConfig(
        entry_script="app.py",
        source_directory="src",
        environment=env
    )

    endpoint_name = "cardio-endpoint"
    try:
        service = AciWebservice(ws, endpoint_name)
        print("Endpoint exists, updating...")
        service.update(models=[model], inference_config=inference_config)
    except:
        print("Endpoint does not exist, creating...")
        deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)
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
    ws = get_workspace()
    print("Connected to workspace:", ws.name)

    compute_target = create_compute_target(ws)
    pipeline, accuracy_output = build_pipeline(ws, compute_target)
    experiment = Experiment(ws, "cardio-pipeline")
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted. Run ID:", pipeline_run.id)
    pipeline_run.wait_for_completion(show=True)

    # Download accuracy output
    pipeline_run.download_file(name=accuracy_output.name, output_file_path="accuracy.txt")
    if os.path.exists("accuracy.txt"):
        with open("accuracy.txt", "r") as f:
            accuracy = float(f.read().strip())
        print(f"Retrieved accuracy: {accuracy}")
        if accuracy > 0.8:
            deploy_model(ws)
        else:
            print("Model accuracy too low, skipping deployment.")
    else:
        print("Accuracy file not found, deployment skipped.")

if __name__ == "__main__":
    run_pipeline()