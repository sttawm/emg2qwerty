# Dockerfile for Vertex AI Training
# Based on PyTorch official image with CUDA support

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install base system dependencies (cached separately for stability)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (separate layer for easier updates)
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY emg2qwerty/ ./emg2qwerty/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Copy training scripts
COPY train_vertex.sh .
COPY train_fusion.py .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set up entry point
ENTRYPOINT ["./train_vertex.sh"]
