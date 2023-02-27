# Display detections on higher resolution frames

If you are running object detection model eg. MobileNet or Yolo, they usually require smaller frame for inferencing (
eg. `300x300` or `416x416`). Instead of displaying bounding boxes on such small frames, you could also stream higher
resolution frames (eg. `video` output
from [ColorCamera](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/#inputs-and-outputs))
and display bounding boxes on these frames. There are several approaches to achieving that, and in this demo we will go
through them.

### 1. Passthrough

Just using the small inferencing frame. Here we used `passthrough` frame
of [MobileNetDetectionNetwork's output](https://docs.luxonis.com/projects/api/en/latest/components/nodes/mobilenet_detection_network/#inputs-and-outputs)
so bounding boxes are in sync with the frame. Other option would be to stream `preview` frames from `ColorCamera` and
sync on the host (or don't sync at all). `300x300` frame, for reference:

![passthrough](https://user-images.githubusercontent.com/18037362/141348065-3b531aa0-eed9-4364-b5b8-b55424cf306d.png)

### 2. Crop high resolution frame

A simple solution to low resolution frame is to stream high resolution frames (eg. `video` output from `ColorCamera`) to
the host, and draw bounding boxes to it. For bounding boxes to match the frame, `preview` and `video` sizes should have
the same aspect ratio, so `1:1`. In the example, we downscale 4k resolution to `720P`, so maximum resolution
is `720x720`, which is exactly the resolution we used (`camRgb.setVideoSize(720,720)`). We could also use `1080P`
resolution and stream `1080x1080` frames back to the host.

![crop_video](https://user-images.githubusercontent.com/18037362/141347876-25b60d3c-9942-4193-99b8-79dfacc2bdd1.png)

### 3. Stretch the frame

A problem that we often encounter with models is that their aspect ratio is `1:1`, not eg. `16x9` as our camera
resolution. This means that some of the FOV will be lost. In
our [How to maximize FOV](https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/) tutorial we showcased
that changing aspect ratio will preserve the whole aspect ratio of the camera, but it will "squeeze"/"stretch" the
frame, as you can see below.

![squeeze](https://user-images.githubusercontent.com/18037362/141347900-8a9d8269-4eb3-4df9-8bb3-434a24cea68f.png)

### 4. Edit bounding boxes

To avoid stretching the frame (as it can have an affect on NN accuracy), we could also stream full FOV `video` from the
device and do inferencing on `300x300` frames. This would, however, mean that we have to re-calculate bounding boxes to
match with different aspect ratio of the image. This approach **does not preserve the whole aspect ratio**, it only
displays bounding boxes on whole FOV `video` frames.

![edit_bb](https://user-images.githubusercontent.com/18037362/141347853-00a1c5ac-d473-4cf9-a9f5-bdf6271e8ebe.png)

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

```bash
python3 $SCRIPT_NAME.py
```