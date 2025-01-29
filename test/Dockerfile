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
ENTRYPOINT ["bash"]