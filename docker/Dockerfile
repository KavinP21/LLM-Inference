# Instinct in a container. Build with tools/docker/instinct.sh build, or:
#
#   docker build -f docker/Dockerfile --target app -t instinct:app .
#   docker build -f docker/Dockerfile --target nvidia -t instinct:nvidia .
#
# "app" is the harness alone, for a Mac whose models run natively (Docker on
# a Mac can't reach the Apple GPU). "nvidia" adds MLX's CUDA build, for Linux
# or WSL2 with an NVIDIA GPU passed through. See docs/docker.md.

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Tools the agents' commands and the terminal expect: git (the app also
# snapshots the workspace with it before each turn), a shell and the usual
# command-line utilities.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash ca-certificates curl git less procps ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 1000 --create-home --shell /bin/bash instinct \
    && mkdir -p /workspace /state \
    && chown instinct:instinct /workspace /state

FROM base AS app-deps
COPY docker/requirements-app.txt /tmp/requirements-app.txt
RUN pip install -r /tmp/requirements-app.txt

FROM app-deps AS nvidia-deps
COPY tools/nvidia/requirements.txt /tmp/requirements-nvidia.txt
RUN pip install -r /tmp/requirements-nvidia.txt

FROM app-deps AS app
COPY python /opt/instinct/python
COPY examples /opt/instinct/examples
COPY docker/entrypoint.sh /usr/local/bin/instinct-entrypoint
ENV PYTHONPATH=/opt/instinct/python \
    HOME=/state/home \
    SHELL=/bin/bash \
    INSTINCT_MODELS=host
WORKDIR /workspace
USER instinct
EXPOSE 8765
ENTRYPOINT ["instinct-entrypoint"]

FROM nvidia-deps AS nvidia
COPY python /opt/instinct/python
COPY examples /opt/instinct/examples
COPY docker/entrypoint.sh /usr/local/bin/instinct-entrypoint
ENV PYTHONPATH=/opt/instinct/python \
    HOME=/state/home \
    SHELL=/bin/bash \
    INSTINCT_MODELS=nvidia \
    MLX_PTX_CACHE_DIR=/state/cache/mlx-ptx \
    HF_HOME=/state/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
USER instinct
EXPOSE 8765
ENTRYPOINT ["instinct-entrypoint"]
