# Monocular Depth Estimation \w Transfer Learning on DepthAI

This example shows an implementation of [Monocular Depth Estimation with Transfer Learning pretrained MobileNetV2](https://github.com/alinstein/Depth_estimation) on DepthAI.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/149_depth_estimation), which is then converted to OpenVINO IR with required flags and converted to blob.

There are two available blob's for different input sizes:

* `depth_estimation_mbnv2_240x320_openvino_2021.4_6shave`, for 240x320 (H x W) input, which runs with ~15 FPS, and
* `depth_estimation_mbnv2_480x640_openvino_2021.4_6shave`, for 480x640 (H x W) input, which runs with ~5 FPS.

Output of the model is a unnormalized density map of size (INPUT_HEIGHT / 2) x (INPUT_WIDTH / 2). We rescale the input frame to have the same size when showing the demo.

![Image example](https://user-images.githubusercontent.com/18037362/140496170-6e3ad321-7314-40cb-8cc0-f622464aa4bd.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -nn {256x320,480x640}, --nn-choice {256x320,480x640}
                        Choose between 2 neural network models from {240x320,480x640} (the smaller one is default)
```
