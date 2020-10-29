# Training data collecting script

This script allows to create a training dataset using the DepthAI, each entry in it will contain
both left, right, rgb and disparity frames stored.

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
usage: main.py [-h] [-p PATH] [-d] [-nd] [-m TIME] [-af {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path where to store the captured data
  -d, --dirty           Allow the destination path not to be empty
  -nd, --no-debug       Do not display debug output
  -m TIME, --time TIME  Finish execution after X seconds
  -af {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}, --autofocus {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}
                        Set AutoFocus mode of the RGB camera

```

Using the defaults, will store dataset in `data` directory

```
python3 main.py
```
