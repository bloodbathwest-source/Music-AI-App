FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install OS-level deps needed for audio handling and builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-base.txt requirements-backend.txt requirements-frontend.txt /app/

# Upgrade pip and install backend deps by default
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements-backend.txt

# Copy source
COPY . /app

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
