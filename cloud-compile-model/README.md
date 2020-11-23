# Compiling model on cloud

This script allows you to compile the OpenVINO model using our online BlobConverter.

## Contents

There are two demo files:

- __`local.py`__ shows the example conversion request using a local OpenVINO model files (stored in `mobilenet-ssd` directory)
- __`zoo.py`__ shows how to download a model from Intel OpenVINO model zoo (definitions can be found [here](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public), saved as `model.yml`)

## Installation

To use this project, install required packages by running

```
python3 -m pip install -r requirements.txt
```

## Usage

To run these examples, simply run

```
python3 local.py
```

or to use model zoo instead of local model, use

```
python3 zoo.py
```
