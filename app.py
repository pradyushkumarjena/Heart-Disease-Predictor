from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline (saved with joblib in train.py)
model = joblib.load("heart_pipeline.pkl")

# Features must match training data
FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values and convert to float, default to 0 if missing
        input_dict = {}
        for feat in FEATURES:
            value = request.form.get(feat)
            if value is None or value.strip() == '':
                value = 0
            input_dict[feat] = float(value)

        # Convert to DataFrame with proper column names
        input_df = pd.DataFrame([input_dict], columns=FEATURES)

        # Predict using pipeline
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Prepare diagnosis and advice
        if prediction == 1:
            diagnosis = "Heart Disease Detected 💔"
            advice = "Please consult a cardiologist and follow a healthy lifestyle."
        else:
            diagnosis = "No Heart Disease ❤️"
            advice = "Keep a healthy diet and regular exercise to maintain heart health."

        # Render report.html with all input values and results
        return render_template(
            "report.html",
            diagnosis=diagnosis,
            probability=f"{probability*100:.2f}%",
            advice=advice,
            **input_dict  # Pass all fields to template
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
