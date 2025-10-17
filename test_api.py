"""
Simple test script for the Drowsiness Detection API
Run this after starting the Flask server with: python app.py
"""

import requests
import json

# Base URL (change if deploying to production)
BASE_URL = "http://localhost:5000"

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
    
    # Sample data - adjust based on your model's input shape
    sample_data = {
        "values": [0.12, 0.3, -0.2, 0.45, 0.67, 0.89, 0.23, 0.56, 0.34, 0.78, 0.45, 0.12, 0.89, 0.34]
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

