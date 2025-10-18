# 🚗 Drowsiness Detection Backend

A **FastAPI** backend for real-time drowsiness detection using TensorFlow.

## 📁 Project Structure

```
Drowsiness-Detection-Backend/
├── main.py                 # FastAPI application
├── model/
│   └── eye_state_model.h5 # Trained TensorFlow model
├── requirements.txt        # Production dependencies
├── Procfile               # Render deployment config
├── runtime.txt            # Python version for Render
└── test_api.py            # API testing script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Option 1: Run directly
python3 main.py

# Option 2: Run with uvicorn (recommended for development)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server will be available at: http://localhost:8000
```

### 3. Test the API

```bash
# Run test script
python3 test_api.py

# Or test manually
curl http://localhost:8000/
curl http://localhost:8000/health
```

## 📡 API Endpoints

### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "Drowsiness Detection API is running 🚀",
  "model_loaded": true
}
```

### `GET /health`
Health monitoring endpoint

**Response:**
```json
{
  "message": "Healthy",
  "model_loaded": true
}
```

### `POST /predict`
Drowsiness prediction endpoint

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "state": "Open",
  "probability": 0.9234
}
```

- `state`: "Open" (alert) or "Closed" (drowsy)
- `probability`: Confidence score (0-1)

## 📚 Interactive API Documentation

FastAPI provides **automatic interactive documentation**!

Once the server is running, visit:
- **http://localhost:8000/docs** - Swagger UI (test API in browser)
- **http://localhost:8000/redoc** - ReDoc (beautiful documentation)

## 💻 Frontend Integration

```javascript
// Example: React/JavaScript
const API_URL = 'http://localhost:8000';

async function detectDrowsiness(base64Image) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Image })
  });
  
  const result = await response.json();
  console.log('State:', result.state);        // "Open" or "Closed"
  console.log('Probability:', result.probability); // 0.9234
  
  return result;
}
```

## 🌐 Deploy to Render

### 1. Push to GitHub

```bash
git add .
git commit -m "FastAPI drowsiness detection backend"
git push origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com)
2. Create new **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **"Deploy"**

Your API will be live at: `https://your-app-name.onrender.com`

## 📦 Dependencies

- **FastAPI** - Modern, fast web framework
- **Uvicorn** - Lightning-fast ASGI server
- **TensorFlow 2.5.0** - Machine learning model
- **Pillow** - Image processing
- **NumPy** - Numerical operations

## 🔧 Troubleshooting

### Model Not Loading
- Ensure `model/eye_state_model.h5` exists
- Check file is not empty (should be ~1.8 MB)

### Port Already in Use
```bash
# Kill process on port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Module Not Found
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## 🎯 Model Information

- **Input Shape**: (1, 24, 24, 1) - Grayscale images
- **Output**: Binary classification [Closed, Open]
- **Architecture**: CNN with Conv2D layers

## 📝 License

MIT License

---

**Built with FastAPI + TensorFlow** 🚀

