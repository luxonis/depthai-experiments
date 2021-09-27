# Record and Replay

This experiment shows how to save encoded camera streams (color/rectified mono), on the host and then replay these recordings to reconstruct
the whole recorded scene, depth included.

When running `record.py`, it will record streams from all devices and will synchronize all streams across all devices on the host. Recordings
will be saved in the specified folder (with `-p`, by default that folder is `recordings/`). Recordings will be saved as MJPEG (motion JPEG)
files. You can manually use `ffmpeg` to convert these `.mjpeg` recordings to `.mp4`. You can also specify `record.convert_to_mp4(True)` to
automatically convert these recordings to `.mp4` (it requires `ffmpeg` system library and `ffmpy3` python library).

For `replay.py`, we have created a default demo app that runs Mobilenet and sends bounding boxes/spatial coordinates to the host where
it these are displayed.

`Replay` class will send depthai recordings (color, rectified mono frames) back to the device to re-calculate stereo disparity/depth. There are a few things you can
specify when using the `Replay` class:

```
# Resize color frames prior to sending them to the device
replay.set_resize_color((width, height))

# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
# It's set to True by default. Setting it to False will squish the image,
# but will preserve the full FOV
replay.keep_aspect_ratio(False)
```


## Pre-requisites

```
python3 -m pip install -r requirements.txt
```

## Record usage
```
usage: record.py [-h] [-p PATH] [-s [STREAMS]] [-f FPS]

optional arguments:
  -h, --help            show this help message and exit
  -p, --path PATH,    default="recordings/"       Path where to store the captured data
  -s, --save STREAMS, default=["color", "mono"]   Choose which streams to save.
  -f, --fps FPS_NUM,  default=30                  Camera sensor FPS, applied to all cams
```

By default, script will save encoded (jpeg) color frames and depth map.
## Replay usage
```
usage: replay.py -p PATH

optional arguments:
  -p PATH, --path PATH  Path where to store the captured data
```
