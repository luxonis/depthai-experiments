# Pose Estimation Example

This example demonstrates how to run [Human Pose Estimation Network](https://docs.openvinotoolkit.org/latest/omz_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html) using DepthAI.

## Demo

### Camera
[![Human pose estimation](https://user-images.githubusercontent.com/5244214/107493701-35f97100-6b8e-11eb-8b13-02a7a8dbec21.gif)](https://www.youtube.com/watch?v=Py3-dHQymko "Human pose estimation on DepthAI")

### Video file
[![Human pose estimation](https://user-images.githubusercontent.com/5244214/110801736-d3bf8900-827d-11eb-934b-9755978f80d9.gif)](https://www.youtube.com/watch?v=1dp2wJ_OqxI "Human pose estimation on DepthAI")

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (otherwise uses the DepthAI 4K color camera)
```
