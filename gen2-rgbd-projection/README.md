## RGBd projection Demo
This example shows how to align depth to rgb camera frame and project depth points into 3d

### Install Dependencies:
`python3 install_requirements.py`

Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   
PS: This example works only on open3D supported platforms.

### Running Example As-Is:
`python3 main.py --align-depth` - Runs colorized pointcloud visualization

`python3 main.py` - Runs depth image and pointcloud visualization (no color)
