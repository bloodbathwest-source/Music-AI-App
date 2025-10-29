# Dockerfile for Music-AI-App reproducible builds
# Python 3.10-slim with audio libraries

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements-base.txt requirements-backend.txt requirements-frontend.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for both frontend and backend
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-c", "print('Container ready. Use docker-compose or specify a command.')"]
