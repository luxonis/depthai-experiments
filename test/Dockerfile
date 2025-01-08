ARG PYTHON_VERSION
FROM alpine AS arg-check
ARG PYTHON_VERSION
RUN echo ${PYTHON_VERSION}
RUN [ -n "$PYTHON_VERSION" ] || (echo "PYTHON_VERSION build argument is required" && exit 1)

FROM python:${PYTHON_VERSION}
ARG PLATFORM
ARG DAI_VERSION

RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    xorg \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99

COPY /tmp/depthai-experiments .
RUN $CMD="python test/gen3_script_tester.py --environment-variables DEPTHAI_PLATFORM=${PLATFORM} --virtual-display --path ./neural-networks/advanced-examples/object-detection/mobile-object-localizer"
RUN if [ -n "$DAI_VERSION" ]; then CMD="$CMD --depthai-version $DAI_VERSION" fi
RUN CMD="$CMD'"

ENTRYPOINT ["/bin/sh", "-c", "eval $CMD"]