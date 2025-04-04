from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/cardio_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"])
        }
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template("result.html", prediction="At Risk" if prediction == 1 else "No Risk")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)