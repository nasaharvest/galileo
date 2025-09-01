FROM mambaorg/micromamba:latest

WORKDIR /model

RUN micromamba install \
        --name base \
        --yes \
        'python<3.11' \
                gcc \
                gxx \
                pip \
                wget \
                unzip \
                huggingface_hub \
 && /opt/conda/bin/wget https://raw.githubusercontent.com/nasaharvest/galileo/refs/heads/main/requirements.txt \
    -O /model/requirements.txt \
 && micromamba run \
    -n base \
    pip install -r /model/requirements.txt zarr \
 && micromamba run \
    -n base \
    pip cache purge \
 && rm /model/requirements.txt \
 && micromamba clean --all --yes

# Clone latest version of the model code
RUN /opt/conda/bin/wget https://github.com/nasaharvest/galileo/archive/refs/heads/main.zip -O /model/galileo.zip \
 && /opt/conda/bin/unzip /model/galileo.zip \
 && mv galileo-main galileo \
 && rm galileo.zip

# Pull tiny and base models
RUN mkdir /model/galileo/data/models/tiny \
 && cd /model/galileo/data/models/tiny \
 && /opt/conda/bin/wget \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/config.json \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/encoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/second_decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/target_encoder.pt \
 && mkdir /model/galileo/data/models/base \
 && cd /model/galileo/data/models/base \
 && /opt/conda/bin/wget \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/config.json \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/encoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/second_decoder.pt \
    https://huggingface.co/nasaharvest/galileo/resolve/main/models/tiny/target_encoder.pt \
 && printf "from pathlib import Path\n\nNANO = Path('/model/galileo/data/models/nano')\nTINY = Path('/model/galileo/data/models/tiny')\nBASE = Path('/model/galileo/data/models/base')" > /model/galileo/model_paths.py

ENV PYTHONPATH="${PYTHONPATH}:/model/galileo:/model"

# Entry point
CMD ["bash"]
