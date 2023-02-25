[中文文档](README.zh-CN.md)

#  Gaze estimation

This example demonstrates how to run 3 stage inference (3-series, 2-parallel) on DepthAI using Gen2 Pipeline Builder.

[![Gaze Example Demo](https://user-images.githubusercontent.com/5244214/106155937-4fa7bb00-6181-11eb-8c23-21abe12f7fe4.gif)](https://user-images.githubusercontent.com/5244214/106155520-0f483d00-6181-11eb-8b95-a2cb73cc4bac.mp4)


Original OpenVINO demo, on which this example was made, is official [here](https://docs.openvinotoolkit.org/2021.1/omz_demos_gaze_estimation_demo_README.html) from Intel and implemented with the NCS2 with nice diagrams and explanation, [here](https://github.com/LCTyrell/Gaze_pointer_controller), from @LCTyrell.

![graph](https://user-images.githubusercontent.com/32992551/103378235-de4fec00-4a9e-11eb-88b2-621180f7edef.jpeg)

Figure: @LCTyrell

## Pre-requisites

Install requirements:
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
python3 main.py -vid ./demo.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
```
