import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--test_path", type=str)
args = parser.parse_args()

model = joblib.load(args.model_path)
df = pd.read_csv(args.test_path)
X_test = df.drop("target", axis=1)
y_test = df["target"]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))