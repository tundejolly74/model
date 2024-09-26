from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction logic here
    return jsonify({'predicted_sales': '...'})  # Replace with your logic

if __name__ == '__main__':
    app.run()
