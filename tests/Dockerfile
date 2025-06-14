ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

ENV PLATFORM=""
ENV DAI_VERSION=""
ENV DAI_NODES_VERSION=""
ENV PYTHON_VERSION_ENV=""
ENV LOG_LEVEL="INFO"
ENV ROOT_DIR="."
ENV STRICT_MODE="no"

# Install Virtual Display
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    xorg \
    ffmpeg \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

# Copy your project files into the container
# NOTE: If ran inside GH workflow or with HIL --sync-workspace this is not needed
# COPY . /tmp/oak-examples

# Setup Test Command
WORKDIR /tmp/oak-examples

# Define the entrypoint script
ENTRYPOINT ["sh", "-c", "\
    set -e && \
    if [ -z \"$PLATFORM\" ]; then \
    echo 'Error: PLATFORM environment variable is required' && exit 1; \
    fi && \
    echo 'Installing dependencies...' && \
    pip install -r tests/requirements.txt && \
    echo 'Running tests...' && \
    pytest -v -r a --log-cli-level=${LOG_LEVEL} --log-file=out.log --color=yes \
    --root-dir=${ROOT_DIR} \
    --depthai-version=${DAI_VERSION} \
    --depthai-nodes-version=${DAI_NODES_VERSION} \
    --environment-variables=DEPTHAI_PLATFORM=${PLATFORM} \
    --virtual-display \
    --platform=${PLATFORM} \
    --python-version=${PYTHON_VERSION_ENV} \
    --strict-mode=${STRICT_MODE} \
    tests/"]
