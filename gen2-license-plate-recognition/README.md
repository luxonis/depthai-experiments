# Gen2 License Plates Recognition

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder.
First, a license plate is detected on the image and then the cropped license frame is sent to text detection network, 
which tries to decode the license plates texts

> :warning: **This demo is adjusted to detect and recognize Chinese License**! It may not work with other license plates

## Demo

[![Gen2 License Plates recognition](https://user-images.githubusercontent.com/5244214/111067985-158f4000-84c7-11eb-9cea-b276a516342d.gif)](https://www.youtube.com/watch?v=buZOWnL9vm0 "License Plates recognition on DepthAI")

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
python3 main.py -vid ./chineese_traffic.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 