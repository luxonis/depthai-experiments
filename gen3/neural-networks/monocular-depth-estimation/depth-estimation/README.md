# Depth estimation with neural network on DepthAI

This example shows an implementation of neural network models for depth estimation that only take output from one camera on DepthAI. The models are converted to OpenVINO IR with required flags and converted to blob.
  - [FastDepth](https://github.com/dwofk/fast-depth) on DepthAI.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth)
  - [Monocular Depth Estimation with Transfer Learning pretrained MobileNetV2](https://github.com/alinstein/Depth_estimation) on DepthAI. Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/149_depth_estimation)
  - [MegaDepth](https://github.com/zl548/MegaDepth) on DepthAI.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth)

There are five available blob's for different input sizes:

* `fast_depth_256x320_openvino_2021.4_6shave`, for 256x320 (H x W) input, which runs with ~30 FPS.
* `fast_depth_480x640_openvino_2021.4_6shave`, for 480x640 (H x W) input, which runs with ~10 FPS.
* `depth_estimation_mbnv2_240x320_openvino_2021.4_6shave`, for 240x320 (H x W) input, which runs with ~15 FPS.
* `depth_estimation_mbnv2_480x640_openvino_2021.4_6shave`, for 480x640 (H x W) input, which runs with ~5 FPS.
* `megadepth_192x256_openvino_2021.4_6shave`, for 192x256 (H x W) input, which runs with ~5 FPS.

Note that the output of the MBNV2 model is a unnormalized density map of size (INPUT_HEIGHT / 2) x (INPUT_WIDTH / 2). We rescale the input frame to have the same size when showing the demo.

![Image example fast depth](https://user-images.githubusercontent.com/18037362/140495636-0721dea1-7eaf-461e-9a39-23a890513324.gif)
![Image example MBNV2](https://user-images.githubusercontent.com/18037362/140496170-6e3ad321-7314-40cb-8cc0-f622464aa4bd.gif)
![Image example mega depth](../depth-estimation/imgs/example.gif)

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
  -nn, --neural-network {fast_small, fast_large, mbnv_small, mbnv_large, mega}
                        Choose the neural network model used for depth estimation (fast_small is default)
```
