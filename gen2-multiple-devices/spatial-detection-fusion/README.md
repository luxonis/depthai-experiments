# Spatial detection fusion
This example demonstrates multiple Luxonis OAK cameras tracking objects and showing their position in bird's-eye view. 

![](img/demo.gif)
## Controls
| key 			| action
| :---			| :---			|
| `q`			| quit 			|
| `d`			| toggle depth view |


## Usage
> Before you can run this demo you need to calibrate the cameras. Go to [multi-cam-calibration](../multi-cam-calibration) and generate a calibration file for each camera. Make sure that the `calibration_data_dir` in the [`config.py`](config.py) is set correctly.

Run the [`main.py`](main.py) with Python 3.

Camera's position will appear in the bird's-eye view along with its detected objects.

![bird's-eye view](img/birdseye.png)