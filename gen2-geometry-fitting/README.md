# Gen2 Geometry Fitting

This example demonstrates fitting of predefined geometric primitives onto point clouds of objects detected in RGBD images. The device captures depth and color information and runs YOLOv5 on the image data. Detected bounding boxes are used to fit geometry onto subsets of the full point cloud. Possible types of geometry are plane, cuboid, sphere and cylinder, the fitted geometry depends on the YOLO detection label. You can expect up to 5FPS, depending on the geometry. Cylinder fitting is in beta and will take up to 3 seconds. The example was tested on a OAK-D PRO device.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luxonis/depthai-experiments
   cd [example-directory]
   ```
2. It is recommended to run this example in a virtual environment
	 ```bash
   conda create --name myenv
   conda activate myenv
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The example can be run with

```bash
python geometry_fitting.py
```
If you would prefer running this in Jupyter, simply run all cells in `geometry_fitting.ipynb`. 

By default this example does not print out the parameters of detected shapes. If you want this functionality change the variable `verbose` to `True`.