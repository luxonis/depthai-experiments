# MegaDepth on DepthAI

This example shows an implementation of [MegaDepth](https://github.com/zl548/MegaDepth) on DepthAI.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth), which is then converted to OpenVINO IR with required flags and converted to blob.

Currently we only provide one blob:

* `megadepth_192x256_openvino_2021.4_6shave`, for 192x256 (H x W) input, which runs with ~5 FPS.

Output is an depth map of the same size. Note that this experiment is very similar to the FastDepth experiment.

![Image example](imgs/example.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
