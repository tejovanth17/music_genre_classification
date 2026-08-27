import os

# Render dynamic port binding
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Worker configuration (Memory-optimized for Render Free/Starter 512MB RAM)
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = 180
graceful_timeout = 30
keepalive = 5

# Logging to stdout for Render Live Log Stream
accesslog = "-"
errorlog = "-"
loglevel = "info"
