# Gaze estimation

This example demonstrates how to run 3 stage inference on DepthAI using Gen2 Pipeline Builder.

Original OpenVINO demo, on which this example was made, is [here](https://github.com/LCTyrell/Gaze_pointer_controller).

## Demo

[![Gaze Example Demo](https://user-images.githubusercontent.com/5244214/96713680-426c7a80-13a1-11eb-81e6-238e3decb7be.gif)](https://www.youtube.com/watch?v=OzgK5-APxBU)



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
python3 main.py -vid ./demo.py
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 
