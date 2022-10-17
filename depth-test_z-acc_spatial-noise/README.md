# Depth test

## Controls
> The "Point Cloud" window must be focused before using the key commands

| key | action |
| :--- | :--- |
| `q` | quit |
| `f` | fit a plane to the point cloud of the selected camera |
| `s` | save the point clouds to the `point-clouds` folder |
| `v` | toggle plane fit visualization |
| `t` | start the test for the selected camera |
| `1` | select the OAK camera |
| `2` | select the Astra camera |

## Usage
Run `main.py` with Python 3.

Point the camera perpendicular to the target plane (a wall or board with random noise). 

Select the ROI on both cameras by dragging on the image view window.

Select a camera and press the `f` key to fit a plane to the point cloud of the selected region. The horizontal and vertical tilt of the camera will be printed to the console. Try to get those values close to zero.

Measure the distance from the camera to the target plane and update the `camera_wall_distance` in [`config.py`](config.py). You can also change the number of samples taken during the test.

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