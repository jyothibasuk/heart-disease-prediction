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
from azureml.core.runconfig import RunConfiguration
from azureml.data.dataset_factory import FileDatasetFactory

# Configuration
DATA_PATH = "data/heart.csv"  # Path in default datastore
MODEL_PATH = "models/cardio_model.pkl"  # Output path for training step
STORAGE_PATH = "pipeline_outputs"  # Base path in default datastore for outputs

def get_workspace():
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
            max_nodes=1  # Limit to 1 node to minimize cost
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    return compute_target

def build_pipeline(ws, compute_target):
    datastore = ws.get_default_datastore()
    if "heart_data" not in ws.datasets:
        FileDatasetFactory.upload_directory(
            src_dir="../data",
            target=(datastore, "data"),
            overwrite=True,
            show_progress=True
        )
        dataset = Dataset.File.from_files(path=(datastore, DATA_PATH))
        dataset.register(ws, name="heart_data")
    dataset = ws.datasets["heart_data"]

    # Define environment and run configuration
    env = Environment("cardio-env")
    env.python.conda_dependencies.add_conda_package("python=3.9")
    env.python.conda_dependencies.add_pip_package("scikit-learn")
    env.python.conda_dependencies.add_pip_package("pandas")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.python.conda_dependencies.add_pip_package("azureml-dataprep[pandas]")  # For datastore access
    env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    env.register(ws)
    run_config = RunConfiguration()
    run_config.environment = env

    # Define outputs as PipelineData
    train_output = PipelineData("train_output", datastore=datastore)
    test_output = PipelineData("test_output", datastore=datastore)
    model_output = PipelineData("model_output", datastore=datastore)
    accuracy_output = PipelineData("accuracy_output", datastore=datastore)

    # Step 1: Preprocess Data
    preprocess_step = PythonScriptStep(
        name="Preprocess Data",
        script_name="preprocess.py",
        arguments=[
            "--data_path", dataset.as_mount(),
            "--train_path", train_output,
            "--test_path", test_output,
            "--storage_path", STORAGE_PATH
        ],
        outputs=[train_output, test_output],
        source_directory="src",
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    # Step 2: Train Model
    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        arguments=[
            "--train_path", train_output,
            "--model_path", model_output,
            "--storage_path", STORAGE_PATH
        ],
        inputs=[train_output],
        outputs=[model_output],
        source_directory="src",
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
    )

    # Step 3: Evaluate Model
    evaluate_step = PythonScriptStep(
        name="Evaluate Model",
        script_name="evaluate.py",
        arguments=[
            "--model_path", model_output,
            "--test_path", test_output,
            "--output", accuracy_output,
            "--storage_path", STORAGE_PATH
        ],
        inputs=[model_output, test_output],
        outputs=[accuracy_output],
        source_directory="src",
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
    )

    pipeline = Pipeline(workspace=ws, steps=[preprocess_step, train_step, evaluate_step])
    return pipeline, model_output, accuracy_output, train_output, test_output

def deploy_model(ws, model_file_path):
    model = Model.register(
        workspace=ws,
        model_path=model_file_path,
        model_name="cardio-model",
        description="Random Forest model for heart disease prediction"
    )
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
        service.wait_for_deployment(show_output=True, timeout_seconds=900)
    except:
        print("Endpoint does not exist, creating...")
        deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)
        service = Model.deploy(
            workspace=ws,
            name=endpoint_name,
            models=[model],
            inference_config=inference_config,
            deployment_config=deployment_config,
            overwrite=True
        )
        service.wait_for_deployment(show_output=True, timeout_seconds=900)
    
    print("Deployment state:", service.state)
    if service.state != "Healthy":
        try:
            logs = service.get_logs()
            print("Deployment logs:")
            print(logs)
        except Exception as e:
            print(f"Failed to retrieve logs: {str(e)}")
    else:
        print("Scoring URI:", service.scoring_uri)
        print("Authentication key:", service.get_keys()[0])

def run_pipeline():
    ws = get_workspace()
    print("Connected to workspace:", ws.name)

    compute_target = create_compute_target(ws)
    pipeline, model_output, accuracy_output, train_output, test_output = build_pipeline(ws, compute_target)
    experiment = Experiment(ws, "cardio-pipeline")
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted. Run ID:", pipeline_run.id)
    pipeline_run.wait_for_completion(show_output=True)

    # Register datasets from storage
    datastore = ws.get_default_datastore()
    if "heart_data_train" not in ws.datasets:
        train_dataset = Dataset.File.from_files(path=(datastore, f"{STORAGE_PATH}/train.csv"))
        train_dataset.register(ws, name="heart_data_train")
    if "heart_data_test" not in ws.datasets:
        test_dataset = Dataset.File.from_files(path=(datastore, f"{STORAGE_PATH}/test.csv"))
        test_dataset.register(ws, name="heart_data_test")

    # Get step runs
    train_step_run = None
    evaluate_step_run = None
    for step_run in pipeline_run.get_children():
        if step_run.name == "Train Model":
            train_step_run = step_run
        elif step_run.name == "Evaluate Model":
            evaluate_step_run = step_run

    # Download and process accuracy
    if evaluate_step_run:
        try:
            evaluate_step_run.download_file(name="outputs/accuracy.txt", output_file_path="accuracy.txt")
            if os.path.exists("accuracy.txt"):
                with open("accuracy.txt", "r") as f:
                    accuracy = float(f.read().strip())
                print(f"Retrieved accuracy: {accuracy}")
                if accuracy > 0.8:
                    # Download model file from Train Model step
                    if train_step_run:
                        train_step_run.download_file(name="outputs/cardio_model.pkl", output_file_path="cardio_model.pkl")
                        if os.path.exists("cardio_model.pkl"):
                            deploy_model(ws, "cardio_model.pkl")
                        else:
                            print("Model file not found locally after download.")
                    else:
                        print("Train Model step not found in pipeline run.")
                else:
                    print("Model accuracy too low, skipping deployment.")
            else:
                print("Accuracy file not found locally after download.")
        except Exception as e:
            print(f"Error downloading accuracy file: {e}")
    else:
        print("Evaluate Model step not found in pipeline run.")

if __name__ == "__main__":
    run_pipeline()