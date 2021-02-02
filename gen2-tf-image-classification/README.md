# Gen2 Tensorflow Image Classification example

This example demonstrates how to run a neural network created using [TensorFlow Image Classification tutorial](https://www.tensorflow.org/tutorials/images/classification)
(which one of our community members has put together in a single Colab Notebook, even with OpenVINO conversion to .blob)


## Demo

**Please note that the detected flower is not correct, [we're working on fixing the detection results](https://github.com/luxonis/depthai-experiments/issues/55)**

![Pedestrian Re-Identification](https://user-images.githubusercontent.com/5244214/106612249-3039d500-6569-11eb-94c3-7efb4267c53b.gif)

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
