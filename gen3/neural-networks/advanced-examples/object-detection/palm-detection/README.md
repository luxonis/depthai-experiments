Palm detection
================

This example demonstrates the Gen3 Pipeline Builder running 
[Mediapipe Palm Detection Network](https://zoo-rvc4.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3). In this demo DepthAI visualizer is integrated.

## Demo:

![demo](images/palm_detection.gif)
--------------------

## Setup
The experiment can be run using peripheral or standalone mode.

### Peripheral mode
For peripheral mode you first need to install the required packages manually.

#### Installation
```
python3 -m pip install -r requirements.txt
```

#### Running
Run the application using python
```
python3 main.py
```
or using oakctl tool
```
oakctl run-script python3 main.py
```
To see the output in visualizer open browser at http://localhost:8000.

### Standalone mode
All the requirements are installed in virtual environment automatically using `oakctl` tool. Environment is setup according to the `oakapp.toml` file.

#### Connecting to the camera
Connect to the device with command
```
oakctl connect
```
If you have more cameras on your network you can list the available devices using `oakctl list` to obtain the IP adress of the desired camera.

With the obtained IP you can connect to the camera. For example desired camera has IP `192.168.0.10`, connect as
```
oakctl connect 192.168.0.10
```
OR

You can connect to the camera with it's index in the `oakctl list` table. For example if the camera is 3rd in the table, connect as
```
oakctl connect 3
```

#### Running
Run the `oakctl` app from the `palm-detection` directory as
```
oakctl app run .
```
To see the output in visualizer open browser at http://192.168.0.10:8000, if `192.168.0.10` is IP of your camera.


## Using the Visualizer
Once the Visualizer is opened, click on Palm Detection tab and you will see the Palm Detection stream. By clicking on *ANNOTATIONS* tooltip you can turn turn the *Palm Bounding Boxes* on and off.