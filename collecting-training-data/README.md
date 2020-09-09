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
usage: main.py [-h] [-t THRESHOLD] [-p PATH] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Maximum difference between packet timestamps to be considered as synced
  -p PATH, --path PATH  Path where to store the captured data
  -d, --dirty           Allow the destination path not to be empty
```

Using the defaults, will use `0.03` ms threshold and will store dataset in `data` directory

```
python3 main.py
```

## TODOs

Script is now utilizing two assumptions (for simplicity):
- Left packet will arrive before right packet
- disparity_color packet will arrive as the last one

When one of those two conditions is not met, the script will not collect the data