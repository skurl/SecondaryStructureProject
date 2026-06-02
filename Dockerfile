# Build for HPC (x86_64) even from Apple Silicon
# docker buildx build --platform linux/amd64 -t ghcr.io/ORG/IMG:TAG --push .

FROM mambaorg/micromamba:0.15.3

# Let RUN steps see the activated env
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# Where micromamba stores envs
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV CONDA_SUBDIR=linux-64

# CRITICAL for Singularity: ensure Python is on PATH even if ENTRYPOINT is not executed
ENV PATH=/opt/conda/bin:$PATH

# Copy your conda environment file
COPY environment.yml /tmp/environment.yml

# 1) Ensure python & pip exist at /opt/conda/bin
RUN micromamba install -y -p $MAMBA_ROOT_PREFIX -c conda-forge python=3.10 pip && \
    # 2) Install the rest from your env file (no torch/torchvision/torchaudio/other heavy packages here)
    micromamba install -y -p $MAMBA_ROOT_PREFIX -f /tmp/environment.yml --override-channels && \
#    # 3) Install PyTorch CUDA wheels (and other heavier packages) via pip
   $MAMBA_ROOT_PREFIX/bin/python -m pip install --no-cache-dir --upgrade pip && \
   $MAMBA_ROOT_PREFIX/bin/python -m pip install --no-cache-dir \
     --index-url https://download.pytorch.org/whl/cu121 \
     torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    # 4) Clean caches
    micromamba clean --all -y
