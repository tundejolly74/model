from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved Random Forest model
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Extract the features from the input (ensure order matches your training features)
    features = np.array([data['Price'], data['Quantity'], data['Discount'],
                         data['Order_Year'], data['Order_Month'],
                         data['CompetitorPrice'], data['Elasticity'],
                         data['CustomerSegment_Segment_B']])
    
    # Reshape the data to match the format expected by the model
    features = features.reshape(1, -1)
    
    # Make prediction using the Random Forest model
    prediction = rf_model.predict(features)
    
    # Return the prediction result in JSON format
    return jsonify({'predicted_sales': prediction[0]})

# Main entry point to start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
