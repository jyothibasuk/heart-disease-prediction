import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--train_path", type=str)
parser.add_argument("--test_path", type=str)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
os.makedirs("data", exist_ok=True)
train_df.to_csv(args.train_path, index=False)  # e.g., data/train.csv
test_df.to_csv(args.test_path, index=False)    # e.g., data/test.csv
print("Data preprocessing completed.")