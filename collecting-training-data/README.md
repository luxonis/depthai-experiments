[中文文档](README.zh-CN.md)

# Training data collecting script

This script allows to create a training dataset using the DepthAI, each entry in it will contain
both left, right, rgb and disparity frames stored.

## Pre-requisites

```
python3 -m pip install -r requirements.txt
```

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script
```
python3 main.py
```

Optional arguments: 
> Using the defaults, will store dataset in `data` directory
```
usage: main.py [-h] [-t THRESHOLD] [-p PATH] [-d] [-nd] [-m TIME] [-af {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}] [-mf MANUALFOCUS] [-et EXPOSURE_TIME] [-ei EXPOSURE_ISO]

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Maximum difference between packet timestamps to be considered as synced
  -p PATH, --path PATH  Path where to store the captured data
  -d, --dirty           Allow the destination path not to be empty
  -nd, --no-debug       Do not display debug output
  -m TIME, --time TIME  Finish execution after X seconds
  -af {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}, --autofocus {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}
                        Set AutoFocus mode of the RGB camera
  -mf MANUALFOCUS, --manualfocus MANUALFOCUS
                        Set manual focus of the RGB camera [0..255]
  -et EXPOSURE_TIME, --exposure-time EXPOSURE_TIME
                        Set manual exposure time of the RGB camera [1..33000]
  -ei EXPOSURE_ISO, --exposure-iso EXPOSURE_ISO
                        Set manual exposure ISO of the RGB camera [100..1600]
```
