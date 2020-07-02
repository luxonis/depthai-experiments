# Social distancing

Below you can see 3 people in a scene.  DepthAI monitors their 3D positions and alerts whenever they are closer than 2 meters, displaying `Too Close` when they are below this threshold, and also overlaying the distance betwen the people at all times and also the 3D position of each person (in x,y,z in meters) from the camera.

[![COVID-19 Social Distancing with DepthAI](https://i.imgur.com/6cYN5rm.jpg)](https://www.youtube.com/watch?v=cJr4IpGMSLA "DepthAI Social Distancing Proof of Concept")

## Pre-requisites

1. Purchase a DepthAI model (see https://shop.luxonis.com/)
2. Install DepthAI API (see [here](https://docs.luxonis.com/api/) for your operating system/platform)

## Install This Social Distancing Example

```
python3 -m pip install -r requirements.txt
```

## Run The Example

```
python3 main.py
```
