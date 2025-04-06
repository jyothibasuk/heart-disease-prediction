import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--train_path", type=str)  # PipelineData directory
parser.add_argument("--test_path", type=str)   # PipelineData directory
parser.add_argument("--storage_path", type=str)  # Datastore path
args = parser.parse_args()

# Preprocess
df = pd.read_csv(args.data_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Write locally to PipelineData outputs
os.makedirs(args.train_path, exist_ok=True)
os.makedirs(args.test_path, exist_ok=True)
train_file = os.path.join(args.train_path, "train.csv")
test_file = os.path.join(args.test_path, "test.csv")
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

# Upload to default datastore
run = Run.get_context()
datastore = run.experiment.workspace.get_default_datastore()
datastore.upload_files(
    files=[train_file, test_file],
    target_path=args.storage_path,
    overwrite=True,
    show_progress=True
)
print("Data preprocessing completed and files uploaded to storage.")