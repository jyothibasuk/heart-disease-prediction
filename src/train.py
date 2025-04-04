import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

df = pd.read_csv(args.train_path)
X_train = df.drop("target", axis=1)
y_train = df["target"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
joblib.dump(model, args.model_path)
print("Model training completed.")