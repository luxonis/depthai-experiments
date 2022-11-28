# People counting

This demo
uses [person_detection_retail_0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html)
neural network to detect people. The demo also displays the number of people detected on the frame.

By default, it will use images from `/images` folder and change the image every 3 seconds. You can also use color camera
input (instead of images from the host) using `OakCamera()` instead of `OakCamera(replay=...)`.

## Demo

[![image](https://user-images.githubusercontent.com/18037362/119807472-11c26580-bedb-11eb-907a-196b8bb92f28.png)](
https://www.youtube.com/watch?v=_cAP-yHhUN4)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```