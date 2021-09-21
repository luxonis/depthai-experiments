[中文文档](README.zh-CN.md)

# Gen2 Age & Gender recognition

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder.
First, a face is detected on the image and then the cropped face frame is sent to age gender recognition network, which
produces the estimated results

**This demo uses script node** to decode the face detection NN (1st stage NN) results. Script then crops out faces from the original high-res frame (based on face detections) and sends them to the age/gender recognition NN (2nd stage NN). Results of the second stage NN are then sent to the host.

## Demo

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/106005496-954a8200-60b4-11eb-923e-b84df9de9fff.gif)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (conflicts with -cam)
```

To use with a video file, run the script with the following arguments

```
python3 main.py -vid ./input.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 
