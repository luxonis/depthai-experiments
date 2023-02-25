[中文文档](README.zh-CN.md)

# Social distancing

Below you can see 3 people in a scene.  DepthAI monitors their 3D positions and alerts whenever they are closer than 2 meters, displaying `Too Close` when they are below this threshold, and also overlaying the distance betwen the people at all times and also the 3D position of each person (in x,y,z in meters) from the camera.

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

## How it Works

![Social Distancing explanation](https://user-images.githubusercontent.com/32992551/101372410-19c51500-3869-11eb-8af4-f9b4e81a6f78.png)
