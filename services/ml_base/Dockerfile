# Shared base image for all ML services (PyTorch installed ONCE)
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch + shared ML dependencies once
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    transformers==4.47.1 \
    bitsandbytes==0.43.3 \
    accelerate==0.27.2 \
    Pillow==10.1.0 \
    numpy==1.26.2 \
    scipy==1.11.4 \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    gunicorn==21.2.0 \
    requests==2.31.0
