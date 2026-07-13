# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base build stage with common system dependencies.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

ARG COMPUTE=core
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMPUTE=${COMPUTE}

WORKDIR /app

# System deps required by upstream recommenders (lightgbm/cornac/numba/etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy wrapper source.
COPY pyproject.toml ./
COPY mcp_server ./mcp_server
COPY skill ./skill
COPY README.md ./

# ---------------------------------------------------------------------------
# core image: CPU-only upstream + wrapper.
# ---------------------------------------------------------------------------
FROM base AS core
ARG COMPUTE=core
ENV COMPUTE=${COMPUTE}

# Install runtime deps as root, then create an unprivileged app user.
RUN pip install --no-cache-dir "recommenders>=1.2.1,<2" \
    && pip install --no-cache-dir -e . \
    && rm -rf /root/.cache/pip \
    && groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /bin/bash appuser \
    && mkdir -p /app/state \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8080
CMD ["recommenders-mcp"]

# ---------------------------------------------------------------------------
# gpu image: adds tensorflow/torch + CUDA runtime.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS gpu
ARG COMPUTE=gpu
ENV COMPUTE=${COMPUTE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    build-essential \
    libgomp1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY mcp_server ./mcp_server
COPY skill ./skill
COPY README.md ./
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --no-cache-dir "recommenders[gpu]>=1.2.1,<2" \
    && python3.11 -m pip install --no-cache-dir -e . \
    && rm -rf /root/.cache/pip \
    && groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /bin/bash appuser \
    && mkdir -p /app/state \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8080
CMD ["recommenders-mcp"]
