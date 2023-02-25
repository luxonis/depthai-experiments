[中文文档](README.zh-CN.md)

# Stereo on host

> Only API version is available for this example

The `main.py` demonstrates how [stereo pipeline works](https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/#internal-block-diagram-of-stereodepth-node) on the OAK device (using depthai). It rectifies mono frames (receives from the OAK camera) and then
uses `cv2.StereoSGBM` to calculate the disparity on the host. It also colorizes the disparity and shows it to the user.

To run this application, run `python3 main.py`

`disp_to_depth.py` calculates a depth frame from a disparity frame and then compares it (using [SSIM](https://en.wikipedia.org/wiki/Structural_similarity)) to
the depth frame calculated on the OAK camera. Similarity between generated (on OAK) and calculated (on host) depth frames should always be above 99%.

To run this application, run `python3 disp_to_depth.py`

## Installation

```
python3 -m pip install -r requirements.txt
```

## Calibrate camera (if needed)

Your device needs to have calibration stored to work with this example



