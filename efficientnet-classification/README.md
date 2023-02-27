# Efficientnet-b0

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
python3 main.py -cam # Use DepthAI 4K RGB camera as input
python3 main.py -vid [vid path] # Use any of your own MP4 videos as input.
python3 main.py -nd # Print data to the console

# For DepthAI SDK
python3 main.py
```
