#!/bin/sh
set -e

if [ -z "$PLATFORM" ]; then
    echo 'Error: PLATFORM environment variable is required'
    exit 1
fi

echo 'Installing dependencies...'
pip install -r tests/requirements.txt

echo 'Running tests...'
pytest -v -r a --log-cli-level=${LOG_LEVEL} --log-file=out.log --color=yes \
    --root-dir=${ROOT_DIR} \
    --depthai-version=${DAI_VERSION} \
    --depthai-nodes-version=${DAI_NODES_VERSION} \
    --environment-variables=DEPTHAI_PLATFORM=${PLATFORM} \
    --virtual-display \
    --platform=${PLATFORM} \
    --python-version=${PYTHON_VERSION_ENV} \
    --strict-mode=${STRICT_MODE} \
    tests/

exit_code=$?
echo "pytest exit code: $exit_code"
exit $exit_codes
