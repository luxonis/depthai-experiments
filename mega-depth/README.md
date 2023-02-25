##  MegaDepth on DepthAI

This example shows an implementation of [MegaDepth](https://github.com/zl548/MegaDepth) on DepthAI in the Gen2 API system.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth), which is then converted to OpenVINO IR with required flags and converted to blob.

Currently we only provide one blob:

* `megadepth_192x256_openvino_2021.4_6shave`, for 192x256 (H x W) input, which runs with ~5 FPS.

Output is an unnormalized depth map of the same size. Note that this experiment is very similar to the FastDepth experiment, but a different model is used and that's why we provide it in a seperate directory.

![Image example](assets/example.gif)

## Pre-requisites

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py [options]
```
