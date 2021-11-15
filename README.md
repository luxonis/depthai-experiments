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

## MJPEG and JSON streaming ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mjpeg-streaming))

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

This lay the groundwork to make DepthAI [OpenDataCam](https://github.com/opendatacam/opendatacam) compatible.

## Stereo Neural Inference Results Visualizer ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation))

Because there are often application-specific host-side filtering to be done on the stereo neural inference results, and because these calculations are lightweight (i.e. could be done on an ESP32), we leave the triangulation itself to the host.

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## People Counting ([here](https://github.com/luxonis/depthai-experiments/blob/master/people-counter))

This is a basic usage example of megaAI and/or DepthAI (although it doesn't actually use the depth aspect of DepthAI): simply counting people in a scene and logging this count.

So you could use this to make plots over a day of room occupancy.  One could modify this example to show *where* in a room those people were, over time, if desirable.  But for now it just produces a count of people - so the total in view of the camera - over time.

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## People Tracker ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker))

This application counts how many people went upwards / downwards / leftwards / rightwards in the video stream, allowing you to receive an information about how many people went into a room or went through a corridor.

The model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the OpenVIN Model Zoo.

[![Watch the demo](https://user-images.githubusercontent.com/18037362/116413235-56e96e00-a82f-11eb-8007-bfcdb27d015c.gif)](https://www.youtube.com/watch?v=MHmzp--pqUA)

## Recording and reconstruction of the scene ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay))

The `record.py` app syncs all streams (across all devices) and saves them into mjpeg/h265 files, or, in case of depth, into rosbag that can be viewed by the [RealSense Viewer](https://www.intelrealsense.com/sdk-2/#sdk2-tools) (gif below). `Replay` class is used to send the frames back to the device, and it allows reconstruction of the depth perception from two (synced) mono frames.

![depth gif](https://user-images.githubusercontent.com/18037362/141661982-f206ed61-b505-4b17-8673-211a4029754b.gif)

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

## Human-Machine safety ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-machine-safety))

Calculates palm spatial coordinates on the host and calculates spatial distance between the palm and a dangerous object.
If spatial distance is below selected threshold, it will warn the user.

[![Watch the demo](https://user-images.githubusercontent.com/18037362/121198687-a1202f00-c872-11eb-949a-df9f1167494f.gif)](https://www.youtube.com/watch?v=BcjZLaCYGi4)
## Multiple devices per host ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-multiple-devices))

This demo shows how you can use multiple devices per host. The demo will find all devices connected to the host and display an RGB preview from each of them.

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")
