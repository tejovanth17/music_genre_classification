import os

# Dynamic port binding for Render
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Single worker with multi-threading to guarantee < 280MB memory footprint (Render 512MB limit)
workers = 1
threads = 2
worker_class = "gthread"
timeout = 180
graceful_timeout = 30
keepalive = 2

# Automatic memory recycling after 100 requests to prevent RAM accumulation
max_requests = 100
max_requests_jitter = 10

# Logging to stdout for Render real-time logs
accesslog = "-"
errorlog = "-"
loglevel = "info"
