FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

# Install lm-evaluation-harness and its dependencies
RUN pip install --no-cache-dir -e ".[api,ifeval,math,multilingual]" && \
    pip install --no-cache-dir fastapi uvicorn aiofiles httpx

# Apply live-events patch: injects live sample event emission into evaluator.py
RUN python3 /app/patches/apply_patch.py

RUN mkdir -p /workspace/results /workspace/logs

EXPOSE 8096

CMD ["python3", "/app/api/main.py"]
