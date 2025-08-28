# Use the official RunPod Pytorch image as a base
FROM runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV RUNPOD_THROW_OOM_ERRORS=true
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PIP_NO_CACHE_DIR=true
ENV PIP_DISABLE_PIP_VERSION_CHECK=true

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    google-perftools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements-dev.txt

# Copy the source code
COPY ./src /src

# Set the entrypoint
CMD ["/src/start.sh"]

