## RGBd projection Demo
This example shows how to align depth to rgb camera frame and project depth points into 3d

### Install Dependencies:
`python3 install_requirements.py`

Note: `python3 install_requirements.py` also tries to install open3D which is required for this example.
open3D is not supported by all platforms, but is required for pointcloud visualization.

### Running Example As-Is:
`python3 main.py --align-depth` - Runs rgbd image and pointcloud visualization

`python3 main.py` - Runs depth image and pointcloud visualization (no color)

`python3 main.py --no-pcl` - Runs example without pointcloud (e.g. if you cannot install open3d)
