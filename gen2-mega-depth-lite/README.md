## [Gen2] MegaDepth on DepthAI

This example shows an implementation of [MegaDepth](https://github.com/zl548/MegaDepth) on DepthAI in the Gen2 API system.  Blob is created with ONNX from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth), which is then converted to OpenVINO IR with required flags and converted to blob.

Currently we only provide one blob:

* `megadepth_192x256_openvino_2021.4_6shave`, for 192x256 (H x W) input, which runs with ~5 FPS.

Output is an unnormalized depth map of the same size. Note that this experiment is very similar to the FastDepth experiment, but a different model is used and that's why we provide it in a seperate directory.

![Image example](imgs/example.gif)

## Pre-requisites

1. Purchase a DepthAI (or OAK) model (see [shop.luxonis.com](https://shop.luxonis.com/)).

2. Install requirements.
   ```
   python3 -m pip install -r requirements.txt
   ```

## OAK-D-Lite usage with Looking Glass Portrait

1. `python3 main.py`
2. Stand/Sit in a still place in front of the RGB camera and press “S” to capture the image.
3. captured images are saved in rgb_depth subfolder
4. Use this on HoloPlay studio by loading the saved images as RGBD Photo and Video option. 
Link to using HoloPlay Studio [here](https://learn.lookingglassfactory.com/tutorials/getting-started-with-holoplay-studio)

Tip: Use Zoom by scrolling and pan the model to better position it in the Looking Glass Portrait using the mouse. 