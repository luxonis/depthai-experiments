[中文文档](README.zh-CN.md)

# Collision avoidance

This experiment's goal is to detect the vehicles moving towards the camera and alert the user if it can be dangerous pass

## Install

```
python3 -m pip install -r requirements.txt
```

## Run

```
python3 main.py
```

## Background

The calculation process of the collision relies on the depth information 
provided by the OAK-D camera.

Whereas we can operate in 3D coordinates, in fact most of the calculations are being made
in 2D, taking only x (horizontal) and z (depth) into account

The collsion may occur when:
- Car trajectory is pointing towards the camera
- Car speed, and therefore time to impact, is below a threshold

If those two conditions are met, we display a warning message on the preview window.

All of these calculations are being done inside `crash_avoidance.py` file.

We also need to persist information about the object between the frames, so that we actually track
the cars moving on the image, not only detect them, therefore in `tracker.py` there is a code which
assign detected cars on the image to previously detected car positions
