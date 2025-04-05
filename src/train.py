import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)  # Mounted dataset path
parser.add_argument("--model_path", type=str)  # PipelineData directory
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.train_path, "train.csv"))
X_train = df.drop("target", axis=1)
y_train = df["target"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
os.makedirs(args.model_path, exist_ok=True)
joblib.dump(model, os.path.join(args.model_path, "cardio_model.pkl"))
print("Model training completed.")