FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 is needed for some sklearn/onnxruntime builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY router ./router
COPY compare ./compare
COPY scripts ./scripts

EXPOSE 8000 8080

CMD ["gunicorn","-k","uvicorn_worker.UvicornWorker","app.main:app","-b","0.0.0.0:8000","--workers","2","--timeout","30","-c","/app/app/gunicorn_conf.py"]
