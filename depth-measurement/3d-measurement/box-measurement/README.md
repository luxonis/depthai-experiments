# Box measurement

This experiment showcases one possible approach for measuring the size of a box using DepthAI.

![demo](https://github.com/luxonis/depthai-experiments/blob/master/gen2-multiple-devices/box-measurement/img/demo.gif)

_For a more accurate version of multi-cam box measurement check out the [demo here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-multiple-devices/box-measurement)._

## Installation

```
python3 -m pip install -r requirements.txt
```

Running the command above also tries to install open3D which is required for this example.
open3D is not supported by all platforms, but is required for pointcloud visualization. Installing open3D on [Python 3.12 is not yet supported](https://stackoverflow.com/questions/62352767/cant-install-open3d-libraries-errorcould-not-find-a-version-that-satisfies-th).

## Usage

Run the application

```
python3 main.py
```

```
optional arguments:
  -h, --help            show this help message and exit
  -maxd MAX_DIST, --max-dist MAX_DIST
                        maximum distance between camera and object in space in meters
  -mins MIN_BOX_SIZE, --min-box-size MIN_BOX_SIZE
                        minimum box size in cubic meters
```

Run the box_estimation and calibration showcases

```
python3 test_box_estimator.py
python3 test_calibration.py
```

## Examples

|                                                         Outputs                                                         | Inputs                                                                                                              |
| :---------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------ |
| ![example_5_out](https://user-images.githubusercontent.com/47612463/177592137-169290fb-a359-4663-9030-050a661b5196.png) | ![example_5](https://user-images.githubusercontent.com/47612463/177592142-bead0286-8934-4c4f-b14f-12e162ff3330.png) |
| ![example_3_out](https://user-images.githubusercontent.com/47612463/177592144-faba302c-4bf6-42f2-9d32-7f69a4a0db02.png) | ![example_3](https://user-images.githubusercontent.com/47612463/177592146-02c191ae-fde7-4790-98ea-2da3da5579a3.png) |
| ![example_1_out](https://user-images.githubusercontent.com/47612463/177592149-045326d6-cc7b-4751-b34e-0fefd951a3d8.png) | ![example_1](https://user-images.githubusercontent.com/47612463/177592151-3cced47a-9a18-4a15-8ff2-1ecbdecaba7b.png) |

## Idea of the approach

- Get a pointcloud from the depth image.
- Find a ground plane using the RANSAC algorithm.
- Find the box by clustering all the outliers of the plane and finding the biggest one.
- Find the height of the box and project the top box plane on a plane parallel to the ground plane.
- Find a minimal rectangle of the projected points.
- Get the four corners of the rectangle.
- Since the box is a cuboid and parallel to the ground plane, you can easily get the other four points.

## Limitations of the approach

- The region of interest that is taken into consideration has to consist only of ground (a plane) and a box.
- Detection of multiple boxes at once is currently not supported, but can be done as long as the boxes are far enough of each other to be detected as different clusters.
- The box has to lay flat on the ground, as that is assumed by the algorithm.

## Possible improvements

- Create a calibration procedure, where ground is segmented without the box - accurate ground segmentation is crucial for the box height estimation.
- Add a user selectable ROI (for now you can only set maximum distance to objects).
- Add filtering and/or merge multiple consecutive pointclouds into one.
