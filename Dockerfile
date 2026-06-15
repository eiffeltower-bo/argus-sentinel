# syntax=docker/dockerfile:1.7
#
# One parameterized image for argus' peek/track CLI. Same base for CPU and CUDA — the only
# difference is which torch wheel index (and optional face extra) is installed:
#
#   CPU (multi-arch):  docker buildx build --platform linux/amd64,linux/arm64 -t argus:cpu .
#   CUDA (amd64):      docker buildx build --platform linux/amd64 \
#                        --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
#                        --build-arg EXTRAS='[face-gpu]' -t argus:cuda --load .
#                      (torch 2.12.0 CUDA wheels: cu126 or cu130.)
#
# CUDA-on-amd64 gets its CUDA from pip wheels (argus/_onnx.py preloads them), so no heavy
# nvidia/cuda base image is needed. Run the CUDA image with `--gpus all`.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRAS=

########## base — python + the system libs cv2/ffmpeg need at runtime ##########
# Shared by builder (so the in-container test suite can import cv2) and runtime.
FROM python:3.12-slim AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

########## builder — resolve + install deps into a venv ##########
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv

WORKDIR /app
ARG TORCH_INDEX
ARG EXTRAS
# Only what's needed to build+install the argus wheel and its deps (no SSH, no lock).
COPY pyproject.toml README.md ./
COPY argus ./argus

# torch/torchvision first, from the requested wheel index (cpu or cuXXX). The subsequent
# install sees `torch==2.12.0` already satisfied (== matches the +cpu/+cuXXX local version)
# and pulls the rest (ultralytics, opencv, ...) from PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.12.0 torchvision==0.27.0 --index-url "${TORCH_INDEX}" \
 && uv pip install ".${EXTRAS}"

########## runtime — slim image with just the venv and ffmpeg ##########
# base supplies ffmpeg (libx264) for TrackingResult.render() and libgl1/libglib2.0-0 for cv2.
FROM base AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/app \
    MPLCONFIGDIR=/tmp/mpl
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Weights are NOT baked: ultralytics auto-downloads yolo11n/s.pt to the writable WORKDIR (/app)
# on first detector use, where the bare-filename lookup resolves. Smaller image; needs network
# on first call (mount a volume at /app to cache them across container recreations if desired).

RUN useradd --create-home --uid 10001 argus && chown -R argus:argus /app
USER argus

ENTRYPOINT ["argus"]
CMD ["--help"]

########## test — runtime deps + the synthetic pytest suite ##########
FROM builder AS test
RUN --mount=type=cache,target=/root/.cache/uv uv pip install pytest
COPY tests ./tests
ENTRYPOINT []
CMD ["pytest", "-q"]

########## dev — full dev env; pair with docker-compose's bind mount over /app ##########
# argus is re-installed EDITABLE so source changes in the bind-mounted /app reflect live
# (the `argus` console script and `import argus` both resolve to /app/argus). Heavy deps stay
# baked in /opt/venv. See docker-compose.yml + the README "Develop in a container" section.
FROM builder AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pytest && uv pip install -e .
# Keep ultralytics' settings out of the mounted repo.
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl
ENTRYPOINT []
CMD ["sleep", "infinity"]

########## mcp — runtime + the MCP server (streamable HTTP) ##########
# Adds only the MCP server stack on top of runtime (baked weights, ffmpeg, non-root user,
# WORKDIR /app, YOLO_CONFIG_DIR=/app). The `argus-mcp` console script already ships in the
# runtime venv; we just add the `mcp` package (runtime has no uv, so use the venv's pip).
# Inherits TORCH_INDEX/EXTRAS via runtime->builder, so a CUDA build serves on the GPU.
# See docker-compose.yml `mcp` / `mcp-gpu` services.
FROM runtime AS mcp
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install "mcp>=1.9"
USER argus
EXPOSE 8000
ENTRYPOINT ["argus-mcp"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
