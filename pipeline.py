import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def preprocess_data():
    df = pd.read_csv("data/heart.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Data split into train and test sets.")

def train_model():
    df = pd.read_csv("data/train.csv")
    X_train = df.drop("target", axis=1)
    y_train = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/cardio_model.pkl")
    print("Model trained and saved.")

def evaluate_model():
    model = joblib.load("models/cardio_model.pkl")
    df = pd.read_csv("data/test.csv")
    X_test = df.drop("target", axis=1)
    y_test = df["target"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {accuracy:.2f}")
    return accuracy

def run_pipeline():
    preprocess_data()
    train_model()
    evaluate_model()

if __name__ == "__main__":
    run_pipeline()