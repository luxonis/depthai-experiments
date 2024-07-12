# Image quality assessment

This is an example of image quality assessment running directly from OAK RGB camera

The model outputs 4 quality classes
* Clean
* Blur
* Occlusion
* Bright


## Demo

![demo-gif](https://i.imgur.com/LcKM0tK.gif)

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
  -nn NN_PATH, --nn_path NN_PATH
                        select model blob path for inference, defaults to image_quality_assessment_256x256_001 from model zoo
```
