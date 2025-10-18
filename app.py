from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import base64
import io
from PIL import Image
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

def load_cnn_model():
    """Load the TensorFlow CNN model from the model directory"""
    global model
    try:
        # Try to find the model file
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        possible_names = ['eye_state_model.h5', 'model.h5', 'eeg_eye_state_model.h5']
        
        model_path = None
        for name in possible_names:
            path = os.path.join(model_dir, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                model_path = path
                break
        
        if not model_path:
            logger.error(f"Model file not found in {model_dir}")
            logger.info("Please ensure eye_state_model.h5 is in the /model directory")
            return False
        
        # Load the model (TensorFlow 2.8 supports batch_shape)
        logger.info(f"Loading model from: {os.path.basename(model_path)}")
        model = load_model(model_path)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("⚠️  Model failed to load, but server will still start.")
        logger.warning("   /predict endpoint will return errors until model is fixed.")
        return False

# Load model on startup
load_cnn_model()

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
    Expects JSON: { "image": "<base64-encoded-image>" }
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
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "Invalid request. Expected JSON with 'image' key containing base64-encoded image."
            }), 400
        
        # Decode base64 image
        try:
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({
                "error": f"Failed to decode image: {str(e)}"
            }), 400
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 24x24 (model input size)
        img = img.resize((24, 24))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape for model: (1, 24, 24, 1)
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0]
        
        # Get the predicted class and probability
        # Assuming model outputs [prob_closed, prob_open] or similar
        if len(prediction) >= 2:
            # Binary classification with 2 outputs
            prob_open = float(prediction[1])
            prob_closed = float(prediction[0])
            state = "Open" if prob_open > prob_closed else "Closed"
            probability = max(prob_open, prob_closed)
        else:
            # Single output (sigmoid)
            probability = float(prediction[0])
            state = "Closed" if probability > 0.5 else "Open"
        
        return jsonify({
            "state": state,
            "probability": round(probability, 4)
        }), 200
        
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
    # Default to port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

