# QR Code detection

This demo uses [qr_code_detection_384x384](https://github.com/luxonis/depthai-model-zoo/tree/main/models/qr_code_detection_384x384) neural network to detect QR codes.

## Tutorial

We wrote a [Deploying QR code detector model](https://docs.luxonis.com/en/latest/pages/tutorials/deploying-custom-model/#qr-code-detector) tutorial that provides step-by-step details on how to convert, compile and deploy a custom AI model (WeChat QR Code detection model) to the device. Deployment part focuses on how this demo was developed.

## Demo

![demo](https://user-images.githubusercontent.com/18037362/173070218-5a069728-f365-4fa1-869f-ef871b90a7f7.gif)

## Decoding

### API version
Inside the `api/main.py` code you have an option (`DECODE=True`) to also decode the QR code detected.
Decoding is performed on the host using the OpenCV library.

### SDK version
The decoding is enabled by default.

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```