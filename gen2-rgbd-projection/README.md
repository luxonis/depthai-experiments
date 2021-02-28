## RGBd projection Demo
This example shows how to align depth to rgb camera frame   

### Install Dependencies:
`python3 install_requirements.py`

Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   
PS: This example works only on open3D supported platforms.

### Before executing change the calibraion information for better results
`T_neg ,R2_right ,R ,M_right ,M_RGB`

The above five variables in the code define the calibration parameters
And to know how to follow the comments from line 41 in the main.py program.

### Running Example As-Is:
`python3 main.py` - Runs without point cloud visualization

