# Inherits PyTorch + all ML deps from shared ml_base (installed ONCE)
FROM multimodal-detection-ml_base:latest

WORKDIR /app
COPY . .

EXPOSE 5003

CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--timeout", "300", "--workers", "1", "--threads", "2", "app:app"]
