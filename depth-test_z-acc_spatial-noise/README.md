# Depth test

## Controls
> The "Point Cloud" window must be focused before using the key commands

| key | action |
| :--- | :--- |
| `q` | quit |
| `g` | set ground truth based (camera wall distance) on measurement from selected camera |
| `f` | fit a plane to the point cloud of the selected camera |
| `s` | save the point clouds to the `point-clouds` folder |
| `c` | toggle rgb / solid color pointcloud |
| `v` | toggle plane fit visualization |
| `t` | start the test for the selected camera |
| `1` | select the OAK camera |
| `2` | select the Astra camera |

## Command line arguments
- `-c`, `--color`: Use color camera for preview
- `-lr`, `--lrcheck`: Left-rigth check for better handling for occlusions
- `-e`, `--extended`: Closer-in minimum depth, disparity range is doubled
- `-s`, `--subpixel`: Better accuracy for longer distance, fractional disparity 32-levels
- `-ct`, `--confidence_threshold`: 0-255, 255 = low confidence, 0 = high confidence
- `-mr`, `--min_range`: Min range in mm
- `-xr`, `--max_range`: Max range in mm
- `-ms`, `--mono_camera_resolution`: Mono camera resolution ("THE_400_P", "THE_480_P", "THE_720_P", "THE_800_P")
- `-m`, `--median`: Media filter ("MEDIAN_OFF", "KERNEL_3x3", "KERNEL_5x5", "KERNEL_7x7")
- `-n`, `--n_samples`: Number of samples in a single test
- `--astra_intrinsic`: Path to astra intrinsic file (.np file containing 3x3 matrix)

## Usage
Run `main.py` with Python 3.

```
python main.py -lr -s --astra_intrinsic <intrinsic_matrix.npy>
```

Point the camera perpendicular to the target plane (a wall or board with random noise). 

Select the ROI on both cameras by dragging on the image view window.

Select a reference camera (Astra) and press the `g` key to set the ground truth based on the measurement.

Select a camera and press the `f` key to fit a plane to the point cloud of the selected region. The horizontal and vertical tilt of the camera will be printed to the console. Try to get those values close to zero.

When satisfied with the camera position and ROI selection press the `t` key to start the tests. The test results will be printed to the console.

## How it works

Firstly the plane is fitted to the point cloud and a rotation matrix is computed based on the rotation between the camera direction and plane normal. The point cloud is then rotated to isolate the camera pose error.

All outliers below the 0.5% percentile and above the 99.5% percentile are removed.

### Z-Accuracy
For each sample a median value of the following errors ( $e_i$ ) is computed:

$$e_i = {p_i}_z - d$$
where $p_i$ is the $i$-th point in the corrected point cloud and $d$ is the distance from the camera to the target (`camera_wall_distance`).

All median values are averaged together to get the $\text{MedAvr}$

The final result is computed as:
$$\text{Z-accuracy} = \frac{100 * \text{MedAvr}}{d}$$

### Spatial noise
Spatial noise is computed as the root mean square of differences in z value:

$$\text{Spatial-Noise} = \sqrt{\sum_{i=0}^n ({p_i}_z - {u_i}_z) / n}$$

where $p_i$ is the $i$-th point in the corrected point cloud and $u_i$ is the $p_i$ projected on the fitted plane.

The result is averaged over all samples.


## Example results
```
Horizontal tilt: 10.556313454428073° LEFT Vertical tilt: 0.16117350969183297° UP
Testing started ...
Adding measurements .........
=== Results ===
10 measurements
Z accuracy: 7.28% of GT
Spatial noise: 6.05 mm
```


## Installation

### Linux
This was tested on Ubuntu 22.

``` bash
cd ~
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev  libxml2-dev libxmlsec1-dev python3-openssl git
```
Install Pyenv:
``` bash
curl https://pyenv.run | bash
```
Add this to the end of `~/.bashrc`:
``` bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
```
source ~/.bashrc
pyenv install 3.9
pyenv global 3.9
```
Download 'OpenNI-Linux-x64-2.3' from https://www.dropbox.com/sh/ou49febb83m476d/AADqCQuI3agPOdhyuihl0NHMa?dl=0 and extract it into `~/OpenNi`
``` bash
cd ~/OpenNi/OpenNI-Linux-x64-2.3
sudo chmod a+x install.sh
sudo ./install.sh
```
Append to the `~/.bashrc` file
```
source ~/OpenNi/OpenNI-Linux-x64-2.3/OpenNIDevEnvironment
```

```
source ~/.bashrc
cd ~
git clone https://github.com/njezersek/astra-pro-point-cloud

cd ~/astra-pro-point-cloud
pip install -r requirements.txt 
```

This might be unnecessary:
```
sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev
sudo apt-get install cmake cmake-gui
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common
sudo apt-get install libflann1.8 libflann-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libvtk5.10-qt4 libvtk5.10 libvtk5-dev
sudo apt-get install libqhull* libgtest-dev
sudo apt-get install freeglut3-dev pkg-config
sudo apt-get install libxmu-dev libxi-dev
sudo apt-get install mono-complete
sudo apt-get install qt-sdk openjdk-8-jdk openjdk-8-jre
```
The script should work:
```
python main.py
```

### Windows

This was tested on Windows 10.

Download the ["Orbbec Camera Driver for Windows"](https://dl.orbbec3d.com/dist/drivers/win32/astra-win32-driver-4.3.0.20.zip) from [www.orbbec3d.com](https://www.orbbec3d.com/index/download.html) (make sure to click the "more" button to find the download link). 

Install the driver and reboot the computer. After that the the Astra Pro should be visible in the Device Manager under `Orbbec/ORBBEC Depth Sensor`.

Download the ["Orbbec OpenNI SDK for Windows"](https://dl.orbbec3d.com/dist/openni2/v2.3.0.85/Orbbec_OpenNI_v2.3.0.85_windows_release.zip) and extract it. Find the `Win64-Release` folder and copy its contents to `C:\Program Files\Orbbec\OpenNI`.

You can test if the camera is working by running the `NiViewer.exe` from the `OpenNI\tools\NiViewer` folder.

To run the script you need to set the `OPENNI2_REDIST64` environment variable to the `OpenNI\sdk\libs` folder. You can do this by running the following command in the powershell:
```powershell
 $Env:OPENNI2_REDIST64="C:/Program Files/Orbbec/OpenNI/sdk/libs"
```
Or to make it permanent, run the following command in the powershell and restart the terminal:
```powershell
[Environment]::SetEnvironmentVariable("OPENNI2_REDIST64", "C:/Program Files/Orbbec/OpenNI/sdk/libs", "Machine")
```

Install the python dependencies:
```bash
python -m pip install -r requirements.txt
```