ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

ENV PLATFORM=""
ENV DAI_VERSION=""

# Install Virtual Display
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    xorg \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

# Setup Test Command
WORKDIR /tmp/depthai-experiments
ENTRYPOINT ["sh", "-c", "\
    if [ -z \"$PLATFORM\" ]; then \
        echo 'Error: PLATFORM environment variable is required' && \
        exit 1; \
    fi && \
    python -u test/gen3_script_tester.py \
    --environment-variables DEPTHAI_PLATFORM=${PLATFORM} \
    --virtual-display \
    --path ./neural-networks/advanced-examples/object-detection/mobile-object-localizer \
    ${DAI_VERSION:+--depthai-version ${DAI_VERSION}}"]