FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
WORKDIR /opt/app

RUN apt-get update --fix-missing && \
    apt-get install -y wget git && \
    apt-get clean

# Install uv and add to PATH
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml .
COPY . .

# Install dependencies
RUN uv sync
RUN uv add transformers==4.38.0 accelerate==0.27.2 datasets==2.17.1 deepspeed==0.13.2 sentencepiece wandb
RUN uv add setuptools
RUN uv add flash-attn==2.3.4 --no-build-isolation

CMD ["bash"]