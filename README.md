[depthai_experiments中文文档](README.zh-CN.md)

# depthai-experiments
Experimental projects we've done with DepthAI.  

**Experiments** can be anything from "here's some code and it works sometimes" to "this is almost a tutorial".  

The following list isn't exhaustive (as we randomly add experiments and we may forget to update this list):

## [Gen2] Gaze estimation ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-gaze-estimation))

This example demonstrates how to run 3 stage (3-series, 2 parallel) inference on DepthAI using [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136).

[![Gaze Example Demo](https://user-images.githubusercontent.com/5244214/96713680-426c7a80-13a1-11eb-81e6-238e3decb7be.gif)](https://www.youtube.com/watch?v=OzgK5-APxBU)

Origina OpenVINO demo, on which this example was made, is [here](https://github.com/LCTyrell/Gaze_pointer_controller)

## [Gen2] Subpixel and LR-Check Disparity Depth ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-camera-demo))

This example shows how to do Subpixel, LR-Check or Extended Disparity, and also how to project these measurements into a point cloud for visualization.  This uses the [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136).

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)
![image](https://user-images.githubusercontent.com/32992551/99454680-fea75b00-28e3-11eb-80bc-2004016d75e2.png)

## [Gen2] Age Gender ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender#gen2-age--gender-recognition))

This shows a simple two-stage neural inference example, doing face detection and then age/gender estimation based on the face.

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/106005496-954a8200-60b4-11eb-923e-b84df9de9fff.gif)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## [Gen2] Text Detection + Optical Character Recognition (OCR) Pipeline ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-ocr#gen2-text-detection--optical-character-recognition-ocr-pipeline))

This pipeline implements text detection (EAST) followed by optical character recognition of the detected text. 

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749667-f6315900-5f00-11eb-92bd-a297590adedc.png)](https://www.youtube.com/watch?v=YWIZYeixQjc "Gen2 OCR Pipeline")

## [Gen2] Pedestrian Reidentification ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification))

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder to identify and re-identify pedestrians with unique IDs.

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).

## COVID-19 Mask / No-Mask Detector ([here](https://github.com/luxonis/depthai-experiments/blob/master/coronamask))

This project shows you how to run the COVID-19 mask/no-mask object detector which was trained [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/90733159-74436100-e2cc-11ea-8fb6-d4be937d90e5.gif)](https://photos.app.goo.gl/mJZ8TdWoNatHzW4x7 "COVID-19 mask detection")

## Social Distancing Example ([here](https://github.com/luxonis/depthai-experiments/tree/master/social-distancing))
Since DepthAI gives the full 3D position of objects in physical space, it's a couple lines of code to make a social-distancing monitor with DepthAI.  So that's what this project is, a quick cut at a social-distancing monitor.

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## Demo-UI ([here](https://github.com/luxonis/depthai-python/tree/gen2_develop/examples))
Application used to demonstrate various capabilities of the DepthAI platform. Contains examples with descriptions,
console outputs and preview windows.

![DemoUI](./demo-ui/preview.png)

## MJPEG and JSON streaming ([here](https://github.com/luxonis/depthai-experiments/tree/master/mjpeg-streaming))

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

This lay the groundwork to make DepthAI [OpenDataCam](https://github.com/opendatacam/opendatacam) compatible.

## Stereo Neural Inference Results Visualizer ([here](https://github.com/luxonis/depthai-experiments/tree/master/triangulation-3D-visualizer))

So because there are often application-specific host-side filtering to be done on the stereo neural inference results, and because these calculations are lightweight (i.e. could be done on an ESP32), we leave the triangulation itself to the host.  If there is interest to do this on DepthAI directly instead, please let us know!

[![Spatial AI](https://user-images.githubusercontent.com/5244214/90748450-7c9e9a80-e2d2-11ea-9e9e-da65b5d9e6f0.gif)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")

## People Counting ([here](https://github.com/luxonis/depthai-experiments/blob/master/people-counter))

This is a basic usage example of megaAI and/or DepthAI (although it doesn't actually use the depth aspect of DepthAI): simply counting people in a scene and logging this count.

So you could use this to make plots over a day of room occupancy.  One could modify this example to show *where* in a room those people were, over time, if desirable.  But for now it just produces a count of people - so the total in view of the camera - over time.

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## People Tracker ([here](https://github.com/luxonis/depthai-experiments/tree/master/people-tracker))

This application counts how many people went upwards / downwards / leftwards / rightwards in the video stream, allowing you to receive an information about how many people went into a room or went through a corridor.

The model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the OpenVIN Model Zoo.  Credits: Adrian Rosebrock, OpenCV People Counter, PyImageSearch, [https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/), accessed on 6 August 2020.

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90752628-ee2d1780-e2d7-11ea-8e48-ca94b02a7674.gif)](https://youtu.be/8RiHkkGKdj0)


## Point Cloud Projection ([here](https://github.com/luxonis/depthai-experiments/blob/master/point-cloud-projection))

This is a simple application which creates rgbd image from `right` and `depth_raw` stream and projects it into point clouds. There is also a interactive point cloud visualizer. (depth_raw with left and rgb will be added soon)

![point cloud visualization](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)


## RGB-D and PCL ([here](https://github.com/luxonis/depthai-experiments/tree/master/pcl-projection-rgb))

This is a simple application which creates rgbd image from `rgb` and `depth` stream and projects it into rgb with depth overlay and point clouds. There is also a interactive point cloud visualizer.

![rgbd](https://media.giphy.com/media/SnW9p4r3feMQGOmayy/giphy.gif)
![rgbd-pcl](https://media.giphy.com/media/UeAlkPpeHaxItO0NJ6/giphy.gif)


## Host-Side WLS Filter ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-wls-filter))

This gives an example of doing host-side WLS filtering using the `rectified_right` and `depth` stream from DepthAI.  

Example running on [BW1092](https://shop.luxonis.com/collections/all/products/bw1092-pre-order) shown below:
![image](https://user-images.githubusercontent.com/32992551/94463964-fc920d00-017a-11eb-9e99-8a023cdc8a72.png)

## Multiple devices per host ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-multiple-devices))

This demo shows how you can use multiple devices per host. The demo will find all devices connected to the host and display an RGB preview from each of them.

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")