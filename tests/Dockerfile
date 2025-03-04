ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

ENV PLATFORM=""
ENV DAI_VERSION=""
ENV PYTHON_VERSION_ENV=""

# Install Virtual Display
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    xorg \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

# Copy your project files into the container
# NOTE: If ran inside GH workflow this is not needed
# COPY . /tmp/depthai-experiments

# Setup Test Command
WORKDIR /tmp/depthai-experiments

# Define the entrypoint script
ENTRYPOINT ["sh", "-c", "\
    if [ -z \"$PLATFORM\" ]; then \
    echo 'Error: PLATFORM environment variable is required' && exit 1; \
    fi && \
    echo 'Installing dependencies...' && \
    pip install -r tests/requirements.txt && \
    echo 'Running tests...' && \
    pytest -v -r a --log-cli-level=INFO --log-file=out.log --color=yes \
    --root-dir=. \
    --depthai-version=${DAI_VERSION} \
    --environment-variables=DEPTHAI_PLATFORM=${PLATFORM} \
    --virtual-display \
    --platform=${PLATFORM} \
    --python-version=${PYTHON_VERSION_ENV} \
    tests/"]