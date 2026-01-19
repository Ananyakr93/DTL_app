# AeroClean Dashboard - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY train_lstm.py .
COPY aqi_data.csv .
COPY *.html .
COPY *.js .
COPY *.css .

# Copy model files if they exist
COPY aqi_lstm_model_improved.keras* ./
COPY scaler_improved.pkl* ./
COPY model_config.pkl* ./

# Create volume for persistent data
VOLUME ["/app/data"]

# Environment variables
ENV AQICN_TOKEN=demo
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["python", "app.py"]
