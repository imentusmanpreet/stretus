FROM python:3.12-slim

WORKDIR /app

# Non-root user matching k8s runAsUser: 1000
RUN useradd -m -u 1000 -s /bin/bash stretus

# Build tools for numpy/pandas + git for datafeed GitHub installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .

RUN chown -R 1000:1000 /app

USER 1000

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]