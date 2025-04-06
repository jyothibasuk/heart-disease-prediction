import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)  # PipelineData directory
parser.add_argument("--test_path", type=str)   # PipelineData directory
parser.add_argument("--output", type=str)
args = parser.parse_args()

model = joblib.load(os.path.join(args.model_path, "cardio_model.pkl"))
df = pd.read_csv(os.path.join(args.test_path, "test.csv"))
X_test = df.drop("target", axis=1)
y_test = df["target"]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "accuracy.txt"), "w") as f:
    f.write(str(accuracy))