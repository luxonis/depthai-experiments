## Gen2 Camera Demo
This example shows how to use the DepthAI/megaAI/OAK cameras in the Gen2 Pipeline Builder over USB.  

Unfiltered subpixel disparity depth results from a [BW1092 board](https://shop.luxonis.com/collections/all/products/bw1092) over USB2:

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)

If you are interested in running over SPI instead (for embedded application), see [here](https://github.com/luxonis/depthai-experiments/tree/depthai-experiments-spi/gen2-spi).  For example the ESP32 onboard the BW1092 can pull the results out over SPI.


### Install Dependencies:
`python3 install_requirements.py`

Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   

### Before executing change the calibraion information for better results
`T_neg ,R2_right ,R ,M_right ,M_RGB`

The above five variables in the code define the calibration parameters
And to know how to follow the comments from line 41 in the main.py program.

### Running Example As-Is:
`python3 main.py` - Runs without point cloud visualization

