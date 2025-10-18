from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import base64
import io
from PIL import Image
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Drowsiness Detection API",
    description="Real-time drowsiness detection using TensorFlow CNN",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the model
model = None

# Request model
class PredictionRequest(BaseModel):
    image: str  # base64 encoded image

# Response model
class PredictionResponse(BaseModel):
    state: str
    probability: float

class HealthResponse(BaseModel):
    message: str
    model_loaded: bool

def load_cnn_model():
    """Load the TensorFlow CNN model from the model directory"""
    global model
    try:
        # Try to find the model file
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        possible_names = ['eye_state_model.h5', 'model.h5']
        
        model_path = None
        for name in possible_names:
            path = os.path.join(model_dir, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                model_path = path
                break
        
        if not model_path:
            logger.error(f"Model file not found in {model_dir}")
            return False
        
        # Load model with custom objects for compatibility
        logger.info(f"Loading model from: {os.path.basename(model_path)}")
        
        import tensorflow as tf
        from tensorflow.keras.layers import InputLayer
        from tensorflow.keras.mixed_precision import Policy as DTypePolicy
        
        # Create custom InputLayer that converts batch_shape to input_shape
        class CustomInputLayer(InputLayer):
            def __init__(self, batch_shape=None, input_shape=None, **kwargs):
                if batch_shape is not None:
                    # Convert batch_shape to input_shape (remove batch dimension)
                    input_shape = batch_shape[1:] if input_shape is None else input_shape
                    kwargs.pop('batch_shape', None)  # Remove batch_shape from kwargs
                super(CustomInputLayer, self).__init__(input_shape=input_shape, **kwargs)
        
        custom_objects = {
            'InputLayer': CustomInputLayer,
            'DTypePolicy': DTypePolicy
        }
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Drowsiness Detection API...")
    load_cnn_model()

@app.get("/", response_model=HealthResponse)
async def home():
    """Health check endpoint"""
    return {
        "message": "Drowsiness Detection API is running 🚀",
        "model_loaded": model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Additional health check endpoint for monitoring"""
    return {
        "message": "Healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint
    Expects JSON: { "image": "<base64-encoded-image>" }
    Returns: { "state": "Open" or "Closed", "probability": float }
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Decode base64 image
        try:
            img_data = base64.b64decode(request.image)
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 24x24 (model input size)
        img = img.resize((24, 24))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Reshape for model: (1, 24, 24, 1)
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0]
        
        # Get the predicted class and probability
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
        
        return {
            "state": state,
            "probability": round(probability, 4)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

