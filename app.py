import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

EXPECTED_FEATURES = [
    'clump_thickness', 'size_uniformity', 'shape_uniformity', 'marginal_adhesion',
    'epithelial_size', 'bare_nucleoli', 'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print("Received features:", features)
    if len(features) != len(EXPECTED_FEATURES):
        return render_template('index.html',
                             prediction_text=f"Error: Expected {len(EXPECTED_FEATURES)} features, but received {len(features)} features.")
    final_features = np.array([features])
    try:
        final_features = scaler.transform(final_features)
    except ValueError as e:
        return render_template('index.html', prediction_text=f"Error in scaling: {str(e)}")
    prediction = model.predict(final_features)
    print("Scaled features:", final_features)
    print("Prediction:", prediction)
    output = prediction[0]
    if output == 0:
        return render_template('index.html',
                             prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER')
    else:
        return render_template('index.html',
                             prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    values = list(data.values())
    if len(values) != len(EXPECTED_FEATURES):
        return jsonify({"error": f"Expected {len(EXPECTED_FEATURES)} features, but received {len(values)} features."})
    final_features = np.array([values])
    try:
        final_features = scaler.transform(final_features)
    except ValueError as e:
        return jsonify({"error": f"Error in scaling: {str(e)}"})
    prediction = model.predict(final_features)
    output = prediction[0]
    return jsonify({"prediction": int(output)})

if __name__ == "__main__":
    app.run(debug=True)