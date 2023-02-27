##  FastDepth on DepthAI

This example shows an implementation of [FastDepth](https://github.com/dwofk/fast-depth) on DepthAI in the Gen2 API system.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth), which is then converted to OpenVINO IR with required flags and converted to blob.

There are two available blob's for different input sizes:

* `fast_depth_256x320_openvino_2021.4_6shave`, for 256x320 (H x W) input, which runs with ~30 FPS, and
* `fast_depth_480x640_openvino_2021.4_6shave`, for 480x640 (H x W) input, which runs with ~10 FPS.

![Image example](https://user-images.githubusercontent.com/18037362/140495636-0721dea1-7eaf-461e-9a39-23a890513324.gif)

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
# For DepthAI API
python3 main.py [options]

# For DepthAI SDK
python3 main.py
```

Options:

* `-w, --width`: Select width of the inference model. Default: *`320`*. Possible: `320` or `640`.

