# base on cuda ubuntu image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# set working directory
WORKDIR /app

# install dependencies
RUN apt update && apt install -y \
    build-essential \
    git \
    gnupg2 \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libopencv-dev \
    lsb-release \
    mesa-common-dev \
    mesa-utils \
    nano \
    python3 \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    tree \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install nsight systems cli
RUN apt update && \
    apt install -y --no-install-recommends gnupg && \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/os-release; echo "$VERSION_ID" | tr -d .)/$(dpkg --print-architecture) /" | \
        tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt update && \
    apt install -y nsight-systems-cli
