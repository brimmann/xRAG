FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
ENV PATH=/usr/local/bin:$PATH
WORKDIR /opt/app

# Add these lines to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update --fix-missing && \
    apt-get install -y wget git python3.9 python3.9-dev python3-pip && \
    apt-get clean

# Ensure python and pip point to Python 3.9
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install --upgrade pip
RUN pip install torch==2.1.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers==4.38.0 accelerate==0.27.2 datasets==2.17.1 deepspeed==0.13.2 sentencepiece wandb
RUN pip install flash-attn==2.3.4 --no-build-isolation
CMD ["bash"]