# The Resolve agent service: the graph behind FastAPI. No key in the image; everything comes
# from the environment at run time (see compose.yaml).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    HF_HOME=/app/var/hf TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
# CPU-only torch keeps the image far smaller than the default CUDA wheels.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY configs ./configs
# data/ (knowledge base, tickets, facts) is mounted read-only by compose; var/ holds checkpoints and model caches.
RUN mkdir -p /app/data /app/var

EXPOSE 8200
CMD ["uvicorn", "resolve.api:app", "--host", "0.0.0.0", "--port", "8200"]
