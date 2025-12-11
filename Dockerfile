FROM python:3.12-slim

WORKDIR /model

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    unzip \
    curl \
    git \
    gdal-bin \
    libgdal-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Clone latest version of the model code
RUN wget https://github.com/nasaharvest/galileo/archive/refs/heads/main.zip -O /model/galileo.zip \
 && unzip /model/galileo.zip \
 && mv galileo-main galileo \
 && rm galileo.zip

WORKDIR /model/galileo

# NOTE: After PR is merged, this will use pyproject.toml from the repo
# For now, copy local files for testing
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --no-dev --frozen

# Pull tiny and base models
RUN mkdir -p data/models/tiny \
 && cd data/models/tiny \
 && wget \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/config.json \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/encoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/second_decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/target_encoder.pt \
 && mkdir -p ../base \
 && cd ../base \
 && wget \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/config.json \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/encoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/second_decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/target_encoder.pt \
 && cd /model/galileo \
 && printf "from pathlib import Path\n\nNANO = Path('/model/galileo/data/models/nano')\nTINY = Path('/model/galileo/data/models/tiny')\nBASE = Path('/model/galileo/data/models/base')" > model_paths.py

ENV PYTHONPATH="/model/galileo:/model"

# Entry point
CMD ["bash"]
