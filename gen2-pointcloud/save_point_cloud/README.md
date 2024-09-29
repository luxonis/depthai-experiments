## Save colored point cloud demo

Capture and display color and depth aligned, plus the rectified stereo images.
Press 'q' on any image window to exit.

Before terminating, the program will save:
- point cloud file in .pcd format
- depth image as 16-bit .png
- rgb image as .png
- rectified Left stereo image as .png
- rectified Right stereo image as .png
- intrinsics used to compute the point cloud as .json
- full camera calibration info as .json

## Usage:

```bash
python main.py
```

data will be saved only when quitting the viewer.

## Optional, view the point cloud with open3d:

```bash
python o3d_view_pcd.py *.pcd
```

