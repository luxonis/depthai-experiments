# Record and Replay

These tools allow you to record encoded and synced camera streams and replay them, including reconstructing the stereo depth perception.

### Recording

When running `record.py`, it will record encoded streams from all devices and will synchronize all streams across all devices on the host. Recordings will be saved in the specified folder (with `-p`, by default that folder is `recordings/`). Recordings will be saved as:
- By default: MJPEG (motion JPEG) files or H265, depending on the quality of the recording You can manually use `ffmpeg` to convert these `.mjpeg` recordings to `.mp4`
- If [PyAv](https://github.com/PyAV-Org/PyAV) is installed: It will save encoded streames directly into `.mp4` containers. Install PyAv with `python3 -mpip install av`. This will allow you to watch videos with a standard video player. More [info here](../../container-encoding).
- If depth is enabled: Program will save depth into rosbag (`.bag`), which you can open with [RealSense Viewer](https://www.intelrealsense.com/sdk-2/#sdk2-tools) (image below)
- If `-mcap` is enabled, depthai-record will record selected streams into [mcap file](https://github.com/foxglove/mcap) and can be viewed with [Foxglove studio](https://foxglove.dev/). Depth is converted to pointcloud on the host before being saved. Standalone Foxglove studio streaming demo can be [found here](../foxglove/).

![depth gif](https://user-images.githubusercontent.com/18037362/141661982-f206ed61-b505-4b17-8673-211a4029754b.gif)

#### Record usage

Navigate to `api` directory:
```bash
cd ./api
```

Then run the script:
```
python record.py [arguments]
```

**Optional arguments:**

- `-p / --path`: Folder path where recordings will be saved. Default: `recordings/`.
- `-save / --save`: Choose which streams to save. Currently supported: `color`, `left`, `right`, `disparity`, `depth` (.bag or .mcap), `pointcloud` (.mcap)
- `-f / --fps`: Camera sensor FPS, applied to all cameras
- `-q / --quality`: Selects the quality of the encoded streams that are being recording. It can either be `BEST` (lossless encoding), `HIGH`, `MEDIUM` or `LOW`. More information regarding **file sizes and quality of recordings** can be [found here](encoding_quality/README.md). Default: `HIGH`. If integer 0..100 is used, MJPEG encoding will be used and the MJPEG quality will be set to the value specified.
- `-fc / --frame_cnt`: Number of frames to record. App will record until it's stopped (CTRL+C) by default. If you select eg. `-fc 300 --fps 30`, recording will be of 300 frames (of each stream), for a total of 10 seconds.
- `-tl / --timelapse`: Number of seconds between saved frames, which is used for timelapse recording. By default, timelapse is disabled.
- `-mcap / --mcap`: Record all streams into the .mcap file, so it can be viewed with [Foxglove Studio](https://foxglove.dev/)

### Replaying

`replay.py` is a demo script that runs Spatial MobileNet network. It will reconstruct stereo depth perception, which will allow it to calculate spatial coordinates as well.

#### Replay usage

`Replay` class (from `libraries/depthai_replay.py`) will read `recordings` and send recorded and synced frames back to the device to reconstruct the stereo depth perception.

There are a few things you can specify when using the `Replay` class:

```pyhton
# First initialize the Replay object, passing path to the depthai_recording
replay = Replay(path)

# Resize color frames prior to sending them to the device
replay.set_resize_color((width, height))

# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
# It's set to True by default. Setting it to False will squish the image,
# but will preserve the full FOV
replay.keep_aspect_ratio(False)

# Don't read/stream recorded disparity
replay.disable_stream("disparity", disable_reading=True)
# Read but don't stream recorded depth
replay.disable_stream("depth")
```
#### Replay usage
```
usage: replay.py -p PATH

optional arguments:
  -p PATH, --path PATH  Path where to store the captured data
```

## Pre-requisites

```
python3 -m pip install -r requirements.txt
```
