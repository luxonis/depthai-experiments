[中文文档](README.zh-CN.md)

# Gen2 Frames Sync

This example demonstrates how to synchronize incoming frames using their sequence numbers, from all 3 on-device cameras. 
This allows displaying frames taken in exact same moment. For a better visualization, all 3 streams are horizontally stacked:  
left + rgb + right.

## Demo

Image captured running with `python3 main.py -d`

![demo](https://user-images.githubusercontent.com/60824841/125710596-f490d5f5-49c5-41b9-a318-d62046450665.png)


## Sync multiple devices

If you would like to sync multiple streams (color/left/right) across multiple devices, you should use `sync-multiple-devices.py` script.
Since timestamp is synced with the host, all devices get the same timestamp thus timestamps are in the script to sync across multiple devices.

![demo](https://user-images.githubusercontent.com/18037362/130965049-0d315888-1ff4-4455-b5ec-668d33e6f051.png)

As you can see, bottom color image is 16ms behind other frames. Since we were using 30FPS, time difference below 16.6 are considered in sync
(`1/FPS => 33.3ms/2 => 16.6`).

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
usage: python3 main.py [-h] [-f FPS] [-d] [-v] [-t]

optional arguments:
  -h, --help           show this help message and exit
  -f FPS, --fps FPS    Camera sensor FPS, applied to all cams
  -d, --draw           Draw on frames the sequence number and timestamp
  -v, --verbose        Verbose, -vv for more verbosity
  -t, --dev_timestamp  Get device timestamps, not synced to host. For debug

Press C to capture a set of frames.
```

### Output with logging enabled

`python3 main.py -d -f 30 -vv`

```
   Seq  Left_tstamp  RGB-Left  Right-Left  Dropped
   num    [seconds]  diff[ms]    diff[ms]  delta
     0     0.055592    -0.785       0.017
     1     0.088907    -0.771       0.011
     2     0.122222    -0.758       0.009
     3     0.155537    -0.745       0.010
     4     0.188852    -0.340       0.011
     5     0.222167    -0.326       0.011
     6     0.255482     0.063       0.010
     7     0.288796     0.078       0.010
     8     0.322111     0.257       0.010
     9     0.355426     0.270       0.009
    10     0.388741     0.246       0.010
    11     0.422056     0.260       0.011
    12     0.455371     0.146       0.009
    13     0.488686     0.160       0.009
    14     0.522000     0.046       0.010
    15     0.555315     0.061       0.010
    16     0.588644    -0.015       0.010
    17     0.621945     0.020       0.011
    18     0.655267    -0.011       0.008
    19     0.688575     0.019       0.010
    20     0.721890     0.018       0.009
    21     0.755205     0.033       0.009
    22     0.788520     0.037       0.012
    23     0.821834     0.050       0.010
    24     0.855149     0.048       0.009
    25     0.888463     0.063       0.010
    26     0.921778     0.055       0.012
...
```
