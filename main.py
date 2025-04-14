import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# Load the trained model
model = joblib.load('crop_recommendation_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    final_input = pd.DataFrame([input_data], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(final_input)[0]
    return render_template('index.html',
                           prediction_text=f'Recommended Crop: {prediction}',
                           values=request.form)

if __name__ == '__main__':
    app.run(debug=True)
