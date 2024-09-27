from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the saved models
with open('linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Linear Regression Model API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    
    # Extract features from the JSON request
    features = np.array([data['Price'], data['Quantity'], data['Discount'],
                         data['Order_Year'], data['Order_Month'],
                         data['CompetitorPrice'], data['Elasticity'],
                         data['CustomerSegment_Segment_B']])
    
    # Reshape features for prediction
    features = features.reshape(1, -1)
    
    # Make the prediction using the loaded model
    prediction = lr_model.predict(features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_sales': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

