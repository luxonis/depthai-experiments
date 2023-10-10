# Gen2 NANOSAM

[NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) embedder ported to OAK and available through DepthAI Model ZOO. You also need mask decoder ONNX, which you can find in the NanoSAM official README or [here](https://drive.google.com/file/d/1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n/view?usp=drive_link). While embedder runs on the device, the ONNX runs on the CPU. You can also run it on GPU using `onnxruntime-gpu` package.

![output_short](https://github.com/luxonis/depthai-experiments/assets/56075061/94aafd80-f4ac-4fa3-b250-4f50ed6fe3d6)

Two examples are provided:
* `run_keypoints.py` - left click (positive points) and middle click (background points)
* `run_yolov6n.py` - object detection defines the mask samples, middle click to set background points

Note that embedder is trained for an image with 1024 x 1024 input, so it is relatively slow on the OAK (~2 FPS). Because it uses learnable positional encoding, it would have to be re-trained for smaller input shape and consequently faster inference. Please refer to offical repository for training information and license.

## Install Requirements

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 run_keypoints.py -dec path_to_decoder.onnx
```
