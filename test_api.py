"""
Simple test script for the Drowsiness Detection API
Run this after starting the Flask server with: python app.py
"""

import requests
import json

# Base URL (change if deploying to production)
# FastAPI with uvicorn on port 8000 (or 7860 for Hugging Face)
BASE_URL = "http://localhost:8000"

def test_home():
    """Test the home endpoint"""
    print("\n📍 Testing GET / endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    """Test the health endpoint"""
    print("\n📍 Testing GET /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict():
    """Test the predict endpoint"""
    print("\n📍 Testing POST /predict endpoint...")
    
    # Create a sample 24x24 grayscale image (simulating an eye)
    import numpy as np
    from PIL import Image
    import base64
    import io
    
    # Create a dummy 24x24 grayscale image
    dummy_img = np.random.randint(0, 255, (24, 24), dtype=np.uint8)
    img = Image.fromarray(dummy_img, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    sample_data = {
        "image": img_base64
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_invalid_request():
    """Test with invalid request"""
    print("\n📍 Testing POST /predict with invalid data...")
    
    invalid_data = {
        "invalid_key": "invalid_value"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n📍 Testing POST /predict with invalid base64...")
    
    invalid_base64 = {
        "image": "not-a-valid-base64-image"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_base64,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("🚀 Starting API Tests...")
    print(f"Base URL: {BASE_URL}")
    print("-" * 50)
    
    try:
        test_home()
        test_health()
        test_predict()
        test_invalid_request()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("Make sure the Flask server is running: python app.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

