## Efficientnet-b0

You can read more about the EfficientDet model in [OpenVINO's Docs](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientnet-b0)
This model includes 1000 classes of image classification. You can view these classes in the `classes.py` script.

Instructions on how to compile/Install the model yourself:
```shell
blobconverter -zn efficentnet-b0 -sh 6
```
Specify the number of Shaves to install accordingly. This command requires Blobconverter v0.10.0


## Demo

The demo classifies the animals in the images as Ibex (Mountain Goat) accurately. Also classifies most common objects.

![result](https://user-images.githubusercontent.com/67831664/119170640-2b9a1d80-ba81-11eb-8a3f-a3837af38a73.jpg)

# Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)

```