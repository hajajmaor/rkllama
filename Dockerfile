# Dockerfile for RKLLama
# Improved by Yoann Vanitou <yvanitou@gmail.com>

FROM condaforge/mambaforge:latest

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 wget curl sudo git build-essential \
        ffmpeg libsm6 libxext6 tzdata cmake \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Create conda environment with Python 3.12
RUN conda create -n rkllama python=3.12 -y
ENV CONDA_DEFAULT_ENV=rkllama
ENV PATH=/opt/conda/envs/rkllama/bin:$PATH

# Install RKNPU driver
# RUN cd /tmp \
#     && git clone https://github.com/rockchip-linux/rknpu.git \
#     && cd rknpu \
#     && mkdir -p /usr/lib \
#     && cp -r drivers/linux-aarch64/usr/lib/* /usr/lib/ \
#     && cp -r rknn/rknn_api/librknn_api/lib/* /usr/lib/ \
#     && cp -r rknn/rknn_utils/librknn_utils/lib/* /usr/lib/ \
#     && ldconfig \
#     && cd .. \
#     && rm -rf rknpu
# Install compatible torch version for Python 3.12 (CPU version to avoid CUDA conflicts)
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install RKNN toolkit
RUN cd /tmp \ 
    && git clone https://github.com/airockchip/rknn-llm --depth=1

WORKDIR /tmp/rknn-llm

# Copy RKLLM runtime libraries and headers
RUN cp ./rkllm-runtime/Linux/librkllm_api/aarch64/* /usr/lib/ && \
    cp ./rkllm-runtime/Linux/librkllm_api/include/* /usr/local/include/ && \
    ldconfig

# Build the demo
WORKDIR /tmp/rknn-llm/examples/rkllm_api_demo/deploy
RUN sed -i 's|GCC_COMPILER_PATH=~/opts/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu|GCC_COMPILER_PATH=/usr/bin|' build-linux.sh && \
    sed -i 's|C_COMPILER=${GCC_COMPILER_PATH}-gcc|C_COMPILER=/usr/bin/gcc|' build-linux.sh && \
    sed -i 's|CXX_COMPILER=${GCC_COMPILER_PATH}-g++|CXX_COMPILER=/usr/bin/g++|' build-linux.sh && \
    sed -i 's|STRIP_COMPILER=${GCC_COMPILER_PATH}-strip|STRIP_COMPILER=/usr/bin/strip|' build-linux.sh && \
    bash build-linux.sh

# Copy the built demo to /usr/bin
RUN cp ./build/build_linux_aarch64_Release/llm_demo /usr/bin/rkllm && \
    chmod +x /usr/bin/rkllm

# Set file limits
RUN echo "* soft nofile 16384" >> /etc/security/limits.conf && \
    echo "* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo "root soft nofile 16384" >> /etc/security/limits.conf && \
    echo "root hard nofile 1048576" >> /etc/security/limits.conf

# Skip RKLLM toolkit installation (x86_64 only, not needed for runtime)
# The runtime libraries and demo are already installed
# WORKDIR /tmp/rknn-llm/rkllm-toolkit/packages
# RUN pip install --force-reinstall --no-deps rkllm_toolkit-1.2.2-cp312-cp312-linux_x86_64.whl




WORKDIR /opt/rkllama

# Copy RKLLM runtime library explicitly
# COPY ./lib/librkllmrt.so /usr/lib/
# RUN chmod 755 /usr/lib/librkllmrt.so && ldconfig

COPY ./lib /opt/rkllama/lib
COPY ./src /opt/rkllama/src
# COPY ./models /opt/rkllama/models
COPY requirements.txt README.md LICENSE *.sh *.py /opt/rkllama/

# Install Python dependencies using conda/pip
RUN pip install -r requirements.txt

# Create the rkllama executable
RUN cat <<'EOF' > /usr/local/bin/rkllama && chmod +x /usr/local/bin/rkllama
#!/bin/bash

# Use installation directory
INSTALL_DIR="/opt/rkllama"
CONFIG_DIR="$INSTALL_DIR/config"

# Source configuration if available
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
fi

# Parse arguments to pass along
ARGS=""
PORT_ARG=""
USE_CONDA=false

for arg in "$@"; do
    if [[ "$arg" == "serve" ]]; then
        # Special handling for 'serve' command
        COMMAND="serve"
    elif [[ "$arg" == "--no-conda" ]]; then
        # Handle no-conda flag
        USE_CONDA=false
    elif [[ "$arg" == --port=* ]]; then
        # Extract port argument
        PORT_ARG="$arg"
    else
        # Add all other arguments
        ARGS="$ARGS $arg"
    fi
done

# Build command with all detected options
if [[ -n "$COMMAND" && "$COMMAND" == "serve" ]]; then
    # For 'serve' command, use server.sh
    FINAL_CMD="$INSTALL_DIR/server.sh"

    # Add port if specified
    if [[ -n "$PORT_ARG" ]]; then
        FINAL_CMD="$FINAL_CMD $PORT_ARG"
    fi

    # Add no-conda flag if specified
    if [[ "$USE_CONDA" == false ]]; then
        FINAL_CMD="$FINAL_CMD --no-conda"
    fi

    # Add any remaining args
    FINAL_CMD="$FINAL_CMD $ARGS"
else
    # For all other commands, use client.sh
    FINAL_CMD="$INSTALL_DIR/client.sh"

    # Add port if specified
    if [[ -n "$PORT_ARG" ]]; then
        FINAL_CMD="$FINAL_CMD $PORT_ARG"
    fi

    # Add no-conda flag if specified
    if [ "$USE_CONDA" == false ]; then
        FINAL_CMD="$FINAL_CMD --no-conda"
    fi

    # Add all other arguments
    FINAL_CMD="$FINAL_CMD $ARGS"
fi

# Execute the final command
eval $FINAL_CMD
EOF

EXPOSE 8080

CMD ["/usr/local/bin/rkllama", "serve"]
# If you want to change the port see
# documentation/configuration.md for the INI file settings.
