# On-device Pointcloud NN model

[Click here](https://docs.luxonis.com/en/latest/pages/tutorials/device-pointcloud) for the **tutorial/blog about this demo**.

This demo uses [custom NN model](file:///home/erik/Luxonis/depthai-docs-website/build/html/pages/tutorials/creating-custom-nn-models.html#run-your-own-cv-functions-on-device) approach to run custom logic - depth to pointcloud conversion - on the OAK camera itself.

The model was inspired by Kornia's [depth_to_3d](https://kornia.readthedocs.io/en/latest/geometry.depth.html?highlight=depth_to_3d#kornia.geometry.depth.depth_to_3d) function, but due to the slow performance, it was then built with pytorch.

## Demo

![image](https://user-images.githubusercontent.com/18037362/158055419-5c80d524-3478-49e0-b7b8-099b07dd57fa.png)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```