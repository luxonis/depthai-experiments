# Gen2 NANOSAM

[NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) embedder ported to OAK and available through DepthAI Model ZOO. You also need mask decoder ONNX, which you can find in the NanoSAM official README or [here](https://drive.google.com/file/d/1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n/view?usp=drive_link). While embedder runs on the device, the ONNX runs on the CPU. You can also run it on GPU using `onnxruntime-gpu`package`.

Two examples are provided:
* `run_keypoints.py` - left click (positive points) and right click (background points)
* `run_yolov6n.py` - object detection defines the mask samples

Note that embedder is trained for an image with 1024 x 1024 input, so it is relatively slow on the OAK. Because it uses learnable positional encoding, it would have to be re-trained for smaller input shape and consequently faster inference. Please refer to offical repository for training information and license.

## Install Requirements

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 run_keypoints.py -dec path_to_decoder.onnx
```
