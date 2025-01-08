ARG PYTHON_VERSION
ARG PLATFORM
ARG DAI_VERSION
FROM python:${PYTHON_VERSION}

# Install Virtual Display
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    xorg \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

# Setup Test Command
COPY /tmp/depthai-experiments .
RUN $CMD="python test/gen3_script_tester.py --environment-variables DEPTHAI_PLATFORM=${PLATFORM} --virtual-display --path ./neural-networks/advanced-examples/object-detection/mobile-object-localizer"
RUN if [ -n "$DAI_VERSION" ]; then CMD="$CMD --depthai-version $DAI_VERSION" fi
RUN CMD="$CMD'"

ENTRYPOINT ["/bin/sh", "-c", "eval $CMD"]