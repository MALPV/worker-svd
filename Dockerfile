
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV TERM=xterm-256color

# Install basic system dependencies and Git
RUN apt update -y && apt install -y \
    software-properties-common \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    google-perftools \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt update -y \
    && apt install -y \
    git \
    git-lfs \
    sudo \
    nano \
    aria2 \
    curl \
    wget \
    unzip \
    unrar \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Verify Python version
RUN python --version && pip --version

# Install PyTorch and related packages
RUN pip install --no-cache-dir \
    torch==2.5.0+cu124 \
    torchvision==0.20.0+cu124 \
    torchaudio==2.5.0+cu124 \
    torchtext==0.18.0 \
    torchdata==0.8.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Copy requirements file
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install other AI/ML specific packages
RUN pip install --no-cache-dir \
    xformers==0.0.28.post2

# Clone ComfyUI and custom nodes
RUN git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI && \
    cd /ComfyUI && \
    pip install -r requirements.txt && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

# Download model files
RUN mkdir -p /ComfyUI/models/checkpoints && \
    mkdir -p /ComfyUI/models/vae && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors \
    -d /ComfyUI/models/checkpoints -o svd_xt.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors \
    -d /ComfyUI/models/vae -o vae-ft-mse-840000-ema-pruned.safetensors

# Copy source code and start script
COPY ./src /ComfyUI/src
COPY ./src/start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
