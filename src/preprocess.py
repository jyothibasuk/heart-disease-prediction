import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--train_path", type=str)  # PipelineData directory
parser.add_argument("--test_path", type=str)   # PipelineData directory
args = parser.parse_args()

df = pd.read_csv(args.data_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
os.makedirs(args.train_path, exist_ok=True)
os.makedirs(args.test_path, exist_ok=True)
train_df.to_csv(os.path.join(args.train_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(args.test_path, "test.csv"), index=False)
print("Data preprocessing completed.")