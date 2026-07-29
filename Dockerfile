FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV GRADIO_ANALYTICS_ENABLED=False

# System deps + Python 3.11 (available natively in Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
        git curl ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /home/user

# PyTorch (cu121)
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# All other deps
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

EXPOSE 7865
RUN chmod -R 755 /home/user
CMD ["/bin/bash"]
