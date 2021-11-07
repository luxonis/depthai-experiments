## [Gen2] Monocular Depth Estimation \w Transfer Learning on DepthAI

This example shows an implementation of [Monocular Depth Estimation with Transfer Learning pretrained MobileNetV2](https://github.com/alinstein/Depth_estimation) on DepthAI in the Gen2 API system.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/149_depth_estimation), which is then converted to OpenVINO IR with required flags and converted to blob.

There are two available blob's for different input sizes:

* `depth_estimation_mbnv2_240x320_openvino_2021.4_6shave`, for 240x320 (H x W) input, which runs with ~15 FPS, and
* `depth_estimation_mbnv2_480x640_openvino_2021.4_6shave`, for 480x640 (H x W) input, which runs with ~5 FPS.

Output of the model is a unnormalized density map of size (INPUT_HEIGHT / 2) x (INPUT_WIDTH / 2). We rescale the input frame to have the same size when showing the demo.

![Image example](https://user-images.githubusercontent.com/18037362/140496170-6e3ad321-7314-40cb-8cc0-f622464aa4bd.gif)

## Pre-requisites

1. Purchase a DepthAI (or OAK) model (see [shop.luxonis.com](https://shop.luxonis.com/)).

3. Install requirements.
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
python3 main.py [options]
```

Options:

* `-w, --width`: Select width of the model for inference. Default: `320`. Possible options: `320` or `640`.
