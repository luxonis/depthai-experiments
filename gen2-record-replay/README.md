# Record and Replay

These tools allow you to record encoded and synced camera streams and replay them, including reconstructing the stereo depth perception.

`record.py` and `replay.py` are using [Record](https://docs.luxonis.com/projects/sdk/en/latest/features/recording/) and [Replay](https://docs.luxonis.com/projects/sdk/en/latest/features/replaying/) DepthAI SDK functionalities under the hood.

### Recording

When running `record.py`, it will record encoded streams from all devices and will synchronize all streams across all devices on the host. Recordings will be saved in the specified folder (with `-p`, by default that folder is `recordings/`). Recordings will be saved as:
- By default: MJPEG (motion JPEG) files or H265, depending on the quality of the recording You can manually use `ffmpeg` to convert these `.mjpeg` recordings to `.mp4`. 
  - Make sure to install the `pip install depthai-sdk[record]`, as this will also install the required `av` ([PyAv](https://github.com/PyAV-Org/PyAV)) library for saving encoded frames directly into container. This will allow you to watch videos with a standard video player. More [info here](../gen2-container-encoding).
- If depth is enabled: Program will save depth into rosbag (`.bag`) or DB3.
<!-- - If `-mcap` is enabled, depthai-record will record selected streams into [mcap file](https://github.com/foxglove/mcap) and can be viewed with [Foxglove studio](https://foxglove.dev/). Depth is converted to pointcloud on the host before being saved. Standalone Foxglove studio streaming demo can be [found here](../gen2-foxglove/). -->

![depth gif](https://user-images.githubusercontent.com/18037362/141661982-f206ed61-b505-4b17-8673-211a4029754b.gif)

#### Record usage

```
python record.py [arguments]
```

**Optional arguments:**

- `-p / --path`: Folder path where recordings will be saved. Default: `recordings/`.
- `-save / --save`: Choose which streams to save. Currently supported: `color`, `left`, `right`, `disparity`, `depth` (rosbag or db3)
- `--fps`: Camera sensor FPS, applied to all cameras
- `-q / --quality`: Selects the quality of the encoded streams that are being recording. It can either be `BEST` (lossless encoding), `HIGH`, `MEDIUM` or `LOW`. More information regarding **file sizes and quality of recordings** can be [found here](encoding_quality/README.md). Default: `HIGH`. If integer 0..100 is used, MJPEG encoding will be used and the MJPEG quality will be set to the value specified.
- `-type`: Either `VIDEO` (default), `ROSBAG`, or `DB3`.
- `--disable_preview` - Disable preview output to reduce resource usage. By default, all streams being saved are displayed.
<!-- - `-fc / --frame_cnt`: Number of frames to record. App will record until it's stopped (CTRL+C) by default. If you select eg. `-fc 300 --fps 30`, recording will be of 300 frames (of each stream), for a total of 10 seconds. -->
<!-- - `-tl / --timelapse`: Number of seconds between saved frames, which is used for timelapse recording. By default, timelapse is disabled. -->

### Replaying

`replay.py` is a demo script that runs Spatial MobileNet network. It will reconstruct stereo depth perception (using [DepthAI SDK's Replay](https://docs.luxonis.com/projects/sdk/en/latest/features/replaying/) functionality), which will allow it to calculate spatial coordinates as well.

#### Replay usage

DepthAI SDK's [Replay functionality](https://docs.luxonis.com/projects/sdk/en/latest/features/replaying/) will read `recordings` and send frames back to the device to replay the whole pipeline, including reconstruction of stereo depth perception.

```python
from depthai_sdk import OakCamera
# Here, instead of using one of the public depthai recordings
# https://docs.luxonis.com/projects/sdk/en/latest/features/replaying/#public-depthai-recordings
# We can specify path to our recording, eg. OakCamera(replay='recordings/1-184430102127631200')
with OakCamera(replay='path/to/recording') as oak:
    oak.replay.set_loop(True)
    left = oak.create_camera('CAM_A') # path/to/recording/CAM_A.mp4
    right = oak.create_camera('CAM_C') # path/to/recording/CAM_C.mp4

    # Reconstruct stereo depth from the recording
    stereo = oak.create_stereo(left=left, right=right)

    # Run Spatial object detection on right video stream
    nn = oak.create_nn('yolov7tiny_coco_640x352', right, spatial=stereo)

    oak.visualize(nn) # Show spatial detections visualized on CAM_C video
```
## Pre-requisites

```
python3 -m pip install -r requirements.txt
```
