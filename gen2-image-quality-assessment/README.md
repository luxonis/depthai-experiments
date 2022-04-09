# Image quality assessment

This is an example of image quality assessment running directly from OAK RGB camera

The model outputs 4 quality classes
* Clean
* Blur
* Occlusion
* Bright


## Demo

![demo-gif](https://i.imgur.com/LcKM0tK.gif)

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
usage: main.py [-h] [-nn NN_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -nn NN_PATH, --nn_path NN_PATH
                        select model blob path for inference, defaults to image_quality_assessment_256x256_001 from model zoo
```
