FROM python:3.12-slim

WORKDIR /app

# Non-root user matching k8s runAsUser: 1000
RUN useradd -m -u 1000 -s /bin/bash stretus

# Build tools for numpy/pandas + git for datafeed GitHub installs.
# wget is needed to fetch the TA-Lib C source tarball below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C library — required by the `TA-Lib` Python wrapper.
# Newer ta-lib-python wheels bundle the C lib for some platforms, but installing
# it explicitly guarantees the build works on every arch.
RUN wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz \
    && ldconfig

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .

RUN chown -R 1000:1000 /app

USER 1000

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]