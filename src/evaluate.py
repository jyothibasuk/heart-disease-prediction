import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)  # PipelineData directory
parser.add_argument("--test_path", type=str)   # PipelineData directory
parser.add_argument("--output", type=str)      # PipelineData directory
parser.add_argument("--storage_path", type=str)  # Datastore path
args = parser.parse_args()

# Evaluate
model = joblib.load(os.path.join(args.model_path, "cardio_model.pkl"))
df = pd.read_csv(os.path.join(args.test_path, "test.csv"))
X_test = df.drop("target", axis=1)
y_test = df["target"]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Write locally to PipelineData output
os.makedirs(args.output, exist_ok=True)
accuracy_file = os.path.join(args.output, "accuracy.txt")
with open(accuracy_file, "w") as f:
    f.write(str(accuracy))

# Upload to default datastore
run = Run.get_context()
datastore = run.experiment.workspace.get_default_datastore()
datastore.upload_files(
    files=[accuracy_file],
    target_path=args.storage_path,
    overwrite=True,
    show_progress=True
)
print(f"Accuracy written to {accuracy_file} and uploaded to storage.")