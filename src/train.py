import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)  # PipelineData directory
parser.add_argument("--model_path", type=str)  # PipelineData directory
parser.add_argument("--storage_path", type=str)  # Datastore path
args = parser.parse_args()

# Train
df = pd.read_csv(os.path.join(args.train_path, "train.csv"))
X_train = df.drop("target", axis=1)
y_train = df["target"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Write locally to PipelineData output
os.makedirs(args.model_path, exist_ok=True)
model_file = os.path.join(args.model_path, "cardio_model.pkl")
joblib.dump(model, model_file)

# Upload to default datastore
run = Run.get_context()
datastore = run.experiment.workspace.get_default_datastore()
datastore.upload_files(
    files=[model_file],
    target_path=args.storage_path,
    overwrite=True,
    show_progress=True
)
print("Model training completed and file uploaded to storage.")