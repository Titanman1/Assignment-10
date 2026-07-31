from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Pre-load the ML model
clf = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient_info = request.json
    
    if not patient_info:
        return jsonify({"error": "No input provided"}), 400

    input_df = pd.DataFrame([patient_info])
    pred = clf.predict(input_df)[0]
    
    msg = "Heart Disease Detected" if pred == 1 else "Normal"
    
    return jsonify({"prediction": msg})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
