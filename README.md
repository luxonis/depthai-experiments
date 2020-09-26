# depthai-experiments
Experimental projects we've done with DepthAI.  

**Experiments** can be anything from "here's some code and it works sometimes" to "this is almost a tutorial".  

The following list isn't exhaustive (as we randomly add experiments and we may forget to update this list), but here are some as of June 1st 2020:

## COVID-19 Mask / No-Mask Detector ([here](https://github.com/luxonis/depthai-experiments/blob/master/coronamask/README.md))

This project shows you how to run the COVID-19 mask/no-mask object detector which was trained [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/90733159-74436100-e2cc-11ea-8fb6-d4be937d90e5.gif)](https://photos.app.goo.gl/mJZ8TdWoNatHzW4x7 "COVID-19 mask detection")

## Social Distancing Example ([here](https://github.com/luxonis/depthai-experiments/tree/master/social-distancing))
Since DepthAI gives the full 3D position of objects in physical space, it's a couple lines of code to make a social-distancing monitor with DepthAI.  So that's what this project is, a quick cut at a social-distancing monitor.

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## Demo-UI ([here](https://github.com/luxonis/depthai-experiments/tree/master/demo-ui))
Application used to demonstrate various capabilities of the DepthAI platform. Contains examples with descriptions,
console outputs and preview windows.

![DemoUI](./demo-ui/preview.png)

## MJPEG and JSON streaming ([here](https://github.com/luxonis/depthai-experiments/tree/master/mjpeg-streaming))

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

This lay the groundwork to make DepthAI [OpenDataCam](https://github.com/opendatacam/opendatacam) compatible.

## Stereo Neural Inference Results Visualizer ([here](https://github.com/luxonis/depthai-experiments/tree/master/triangulation-3D-visualizer))

So because there are often application-specific host-side filtering to be done on the stereo neural inference results, and because these calculations are lightweight (i.e. could be done on an ESP32), we leave the triangulation itself to the host.  If there is interest to do this on DepthAI directly instead, please let us know!

[![Spatial AI](https://user-images.githubusercontent.com/5244214/90748450-7c9e9a80-e2d2-11ea-9e9e-da65b5d9e6f0.gif)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")

## People Counting ([here](https://github.com/luxonis/depthai-experiments/blob/master/people-counter/README.md))

This is a basic usage example of megaAI and/or DepthAI (although it doesn't actually use the depth aspect of DepthAI): simply counting people in a scene and logging this count.

So you could use this to make plots over a day of room occupancy.  One could modify this example to show *where* in a room those people were, over time, if desirable.  But for now it just produces a count of people - so the total in view of the camera - over time.

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## People Tracker ([here](https://github.com/luxonis/depthai-experiments/tree/master/people-tracker))

This application counts how many people went upwards / downwards / leftwards / rightwards in the video stream, allowing you to receive an information about how many people went into a room or went through a corridor.

The model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the OpenVIN Model Zoo.  Credits: Adrian Rosebrock, OpenCV People Counter, PyImageSearch, [https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/), accessed on 6 August 2020.

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90752628-ee2d1780-e2d7-11ea-8e48-ca94b02a7674.gif)](https://youtu.be/8RiHkkGKdj0)


## Point Cloud Projection ([here](https://github.com/luxonis/depthai-experiments/tree/pcl_generation/point-cloud-projection))

This is a simple application which creates rgbd image from `right` and `depth_raw` stream and projects it into point clouds. There is also a interactive point cloud visualizer. (depth_raw with left and rgb will be added soon)

![point cloud visualization](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)