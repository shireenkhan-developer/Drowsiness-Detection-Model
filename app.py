from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
model = None

def load_model():
    """Load the TensorFlow model from the model directory"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            logger.info("Please ensure model.h5 is placed in the /model directory")
            return False
        
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Drowsiness Detection API is running 🚀",
        "model_loaded": model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON: { "values": [0.12, 0.3, -0.2, ...] }
    Returns: { "state": "Open" or "Closed", "probability": float }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data or 'values' not in data:
            return jsonify({
                "error": "Invalid request. Expected JSON with 'values' key."
            }), 400
        
        # Extract values
        values = data['values']
        
        if not isinstance(values, list):
            return jsonify({
                "error": "'values' must be a list of numbers."
            }), 400
        
        # Convert to numpy array and reshape for model input
        input_data = np.array(values).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data, verbose=0)
        
        # Get probability (assuming binary classification)
        probability = float(prediction[0][0])
        
        # Determine state based on probability threshold
        # Threshold: > 0.5 = Closed (drowsy), <= 0.5 = Open (alert)
        state = "Closed" if probability > 0.5 else "Open"
        
        return jsonify({
            "state": state,
            "probability": round(probability, 4)
        }), 200
        
    except ValueError as ve:
        return jsonify({
            "error": f"Invalid input data: {str(ve)}"
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Additional health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

