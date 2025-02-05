# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda12.1.0

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

# Environment variables with default values
ENV PROMPT="A cat holding a sign that says hello world" \
    HEIGHT=768 \
    WIDTH=1360 \
    NUM_INFERENCE_STEPS=4 \
    GUIDANCE_SCALE=0.0 \
    MAX_SEQUENCE_LENGTH=256 \
    DISABLE_LOG_STATS=false

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Download and cache the model during build
RUN python3.11 -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')"

# Add src files
ADD src .

CMD python3.11 -u /handler.py
