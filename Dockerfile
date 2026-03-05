# Dockerfile for Vertex AI Training
# Based on PyTorch official image with CUDA support

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY emg2qwerty/ ./emg2qwerty/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy training script
COPY train_vertex.sh .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set up entry point
ENTRYPOINT ["./train_vertex.sh"]
