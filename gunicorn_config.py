# Gunicorn configuration file for Render deployment

# Bind to the port Render provides
bind = "0.0.0.0:10000"

# Worker configuration
workers = 1  # Single worker to save memory on free tier
worker_class = "sync"
threads = 1

# Timeout settings - increase for TensorFlow model prediction
timeout = 120  # 2 minutes for slow predictions
graceful_timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload the app to load model once
preload_app = True

# Memory optimization
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10

