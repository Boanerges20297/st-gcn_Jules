# Dockerfile for ST-GCN Crime Prediction System

# Use a stable, slim Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files to disc
# and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    APP_PORT=5050 \
    FLASK_DEBUG=0

# Install system dependencies required for scientific packages and GeoPandas
# libgdal-dev and build-essential are often needed for compiling wheels or specific C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies globally in the container
# We don't need a venv here because the container IS the environment
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Create necessary directories and set permissions
# These must exist and be writable by the appuser
RUN mkdir -p /app/logs /app/models /app/data /app/outputs /app/static_export/data /opt/screenshot-report_preview/public/data && \
    chown -R appuser:appuser /app

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose the Flask port
EXPOSE 5050

# Healthcheck to ensure the service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5050/api/model-update-status || exit 1

# Start the application
# We use python app.py directly because the app logic (loading models)
# is currently tied to the __main__ block in app.py
CMD ["python", "app.py"]
