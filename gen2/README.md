[depthai_experiments中文文档](README.zh-CN.md)
# depthai-experiments

[![Discord](https://img.shields.io/discord/790680891252932659?label=Discord)](https://discord.gg/luxonis)
[![Forum](https://img.shields.io/badge/Forum-discuss-orange)](https://discuss.luxonis.com/)
[![Docs](https://img.shields.io/badge/Docs-DepthAI-yellow)](https://docs.luxonis.com)

Projects we've done with DepthAI. These can be anything from "here's some code and it works most of the time" to "this is almost a tutorial".

The following list isn't exhaustive (as we randomly add experiments and we may forget to update this list):

## Gaze Estimation ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-gaze-estimation#gen2-gaze-estimation))

[![Gaze Example Demo](https://github.com/luxonis/depthai-experiments/assets/18037362/6c7688e5-30bc-4bed-8455-8b8e9899c5b0)

## Age and Gender Recognition ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender#gen2-age--gender-recognition))

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/106005496-954a8200-60b4-11eb-923e-b84df9de9fff.gif)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## Automated Face-Blurring ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces#gen2-blur-faces-in-real-time))

![Blur Face](https://user-images.githubusercontent.com/18037362/139135932-b907f037-9336-4c42-a479-5715d9693c9c.gif)

## Spatial Calculation - On Host to Show/Explain Math That Happens in OAK-D for the Spatial Location Calculator ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-calc-spatials-on-host#calculate-spatial-coordinates-on-the-host))

![Demo](https://user-images.githubusercontent.com/18037362/146296930-9e7071f5-33b9-45f9-af21-cace7ffffc0f.gif)

## Multi-camera spatial-detection-fusion ([here](./gen2-multiple-devices/spatial-detection-fusion))

![demo](https://github.com/luxonis/depthai-experiments/blob/master/gen2-multiple-devices/spatial-detection-fusion/img/demo.gif?raw=true)

## Stereo Depth from Camera and From Host ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-camera-demo#gen2-camera-demo))

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)

## Automatic JPEG Encoding and Saving Based on AI Results ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-class-saver-jpeg#gen2-class-saver-jpeg))


   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018096-47163c80-67a0-11eb-88f6-c67fb3c2f421.jpg)
  
- `overlay_frame` represents a path to RGB frame with detection overlays (bounding box and label)

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018179-63b27480-67a0-11eb-8423-4fd311a6d860.jpg)

- `cropped_frame` represents a path to cropped RGB frame containing only ROI of the detected object

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018256-7dec5280-67a0-11eb-964e-2cc08b6b75fd.jpg)

An example entries in `dataset.csv` are shown below

```
timestamp,label,left,top,right,bottom,raw_frame,overlay_frame,cropped_frame
16125187249289,bottle,0,126,79,300,data/raw/16125187249289.jpg,data/bottle/16125187249289_overlay.jpg,data/bottle/16125187249289_cropped.jpg
16125187249289,person,71,37,300,297,data/raw/16125187249289.jpg,data/person/16125187249289_overlay.jpg,data/person/16125187249289_cropped.jpg
16125187249653,bottle,0,126,79,300,data/raw/16125187249653.jpg,data/bottle/16125187249653_overlay.jpg,data/bottle/16125187249653_cropped.jpg
16125187249653,person,71,36,300,297,data/raw/16125187249653.jpg,data/person/16125187249653_overlay.jpg,data/person/16125187249653_cropped.jpg
16125187249992,bottle,0,126,80,300,data/raw/16125187249992.jpg,data/bottle/16125187249992_overlay.jpg,data/bottle/16125187249992_cropped.jpg
16125187249992,person,71,37,300,297,data/raw/16125187249992.jpg,data/person/16125187249992_overlay.jpg,data/person/16125187249992_cropped.jpg
16125187250374,person,37,38,300,299,data/raw/16125187250374.jpg,data/person/16125187250374_overlay.jpg,data/person/16125187250374_cropped.jpg
16125187250769,bottle,0,126,79,300,data/raw/16125187250769.jpg,data/bottle/16125187250769_overlay.jpg,data/bottle/16125187250769_cropped.jpg
16125187250769,person,71,36,299,297,data/raw/16125187250769.jpg,data/person/16125187250769_overlay.jpg,data/person/16125187250769_cropped.jpg
16125187251120,bottle,0,126,80,300,data/raw/16125187251120.jpg,data/bottle/16125187251120_overlay.jpg,data/bottle/16125187251120_cropped.jpg
16125187251120,person,77,37,300,298,data/raw/16125187251120.jpg,data/person/16125187251120_overlay.jpg,data/person/16125187251120_cropped.jpg
16125187251492,bottle,0,126,79,300,data/raw/16125187251492.jpg,data/bottle/16125187251492_overlay.jpg,data/bottle/16125187251492_cropped.jpg
16125187251492,person,74,38,300,297,data/raw/16125187251492.jpg,data/person/16125187251492_overlay.jpg,data/person/16125187251492_cropped.jpg
```

## Face Mask Detection ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask))

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/112673778-6a3a9f80-8e65-11eb-9b7b-e352beffe67a.gif)](https://youtu.be/c4KEFG2eR3M "COVID-19 mask detection")

## Crowd Counting ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-crowdcounting#gen2-crowd-counting-with-density-maps-on-depthai))

![Image example](https://raw.githubusercontent.com/luxonis/depthai-experiments/master/gen2-crowdcounting/imgs/example.gif)

## Cumulative Object Counting ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-cumulative-object-counting#cumulative-object-counting))

![cumulative object counting](https://raw.githubusercontent.com/TannerGilbert/Tensorflow-2-Object-Counting/master/doc/cumulative_object_counting.PNG)

## How to Run Customer CV Models On-Device ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-custom-models#demos))

**Concatenate frames**

![Concat frames](https://user-images.githubusercontent.com/18037362/134209980-09c6e2f9-8a26-45d5-a6ad-c31d9e2816e1.png)

**Blur frames**

![Blur frames](https://docs.luxonis.com/en/latest/_images/blur.jpeg)

**Corner detection**

![Laplacian corner detection](https://user-images.githubusercontent.com/18037362/134209951-4e1c7343-a333-4fb6-bdc9-bc86f6dc36b2.jpeg)

## Semantic Segmentation of Depth ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_depth#gen2-deeplabv3-on-depthai---depth-cropping))

![Deeplabv3 Depth GIF](https://user-images.githubusercontent.com/59799831/132396685-c494f21b-8101-4be4-a787-dd382ae6b470.gif)

## Multi-Class Semantic Segmentation ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass#gen2-deeplabv3-multiclass-on-depthai))

![Multi-class Semantic Segmentation](https://raw.githubusercontent.com/luxonis/depthai-experiments/master/gen2-deeplabv3_multiclass/imgs/example.gif)

## Depth-Driven Focus ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-depth-driven-focus#depth-driven-focus))

![Depth driven focus](https://user-images.githubusercontent.com/18037362/144228694-68344fce-8932-4c23-b2f0-601be59184b6.gif)

## Monocular Depth Estimation - Neural Network Based ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-depth-mbnv2))

![Image example](https://user-images.githubusercontent.com/18037362/140496170-6e3ad321-7314-40cb-8cc0-f622464aa4bd.gif)

## Tutorial on How To Display High-Res Object Detections ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-display-detections))

![edit_bb](https://user-images.githubusercontent.com/18037362/141347853-00a1c5ac-d473-4cf9-a9f5-bdf6271e8ebe.png)

## Running EfficientDet Object Detector On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientDet))

[![Watch the demo](https://user-images.githubusercontent.com/18037362/117892266-4c5bb980-b2b0-11eb-9c0c-68f5da6c2759.gif)](https://www.youtube.com/watch?v=UHXWj9TNGrM)

## Running EfficientNet Image Classifier On-Camera ([here]([url](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientnet-classification#efficientnet-b0)))

![result](https://user-images.githubusercontent.com/67831664/119170640-2b9a1d80-ba81-11eb-8a3f-a3837af38a73.jpg)

## Facial Expression (Emotion) Recognition On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-emotion-recognition#gen2-emotion-recognition))

![Demo](https://user-images.githubusercontent.com/18037362/140508779-f9b1465a-8bc1-48e0-8747-80cdb7f2e4fc.png)

## Face Detection On-Camera (libfacedetection) ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection#gen2-face-detection-on-depthai))

![libfacedetection](https://github.com/luxonis/depthai-experiments/blob/master/gen2-face-detection/imgs/example.gif?raw=true)

## Face Recognition On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-recognition#face-recognition))

[![Face recognition](https://user-images.githubusercontent.com/18037362/134054837-eed40899-7c1d-4160-aaf0-1d7c405bb7f4.gif)](https://www.youtube.com/watch?v=HNAeBwNCRek "Face recognition")

## Facial Landmarks On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-facemesh))

![Facial Landmarks](https://github.com/luxonis/depthai-experiments/blob/master/gen2-facemesh/imgs/example.gif?raw=true)

## Fire Detection On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-fire-detection#fire-detection))

![Fire Detection](https://github.com/luxonis/depthai-experiments/blob/master/gen2-fire-detection/images/fire_demo.gif?raw=true)

## Head Posture Detection On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-head-posture-detection#gen2-head-posture-detection))

![Head Pose Detection](https://user-images.githubusercontent.com/18037362/172148301-45adb7ce-3aab-478f-8cad-0c05f349ce50.gif)

## Human-Machine Safety Example On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-machine-safety#gen2-human-machine-safety))

[![Watch the demo](https://user-images.githubusercontent.com/18037362/121198687-a1202f00-c872-11eb-949a-df9f1167494f.gif)](https://www.youtube.com/watch?v=BcjZLaCYGi4)

## Human Skeletal Pose Estimation ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose#gen2-pose-estimation-example))

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/107493701-35f97100-6b8e-11eb-8b13-02a7a8dbec21.gif)](https://www.youtube.com/watch?v=Py3-dHQymko "Human pose estimation on DepthAI")

## LaneNet Lane Segmentation On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-lanenet))

![LaneNet Lane Segmentation](https://github.com/luxonis/depthai-experiments/blob/master/gen2-lanenet/imgs/example.gif?raw=true)

## License Plate Recognition On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-license-plate-recognition#gen2-license-plates-recognition))

[![Gen2 License Plates recognition](https://user-images.githubusercontent.com/5244214/111202991-c62f3980-85c4-11eb-8bce-a3c517abeca1.gif)](https://www.youtube.com/watch?v=tB_-mVVNIro "License Plates recognition on DepthAI")

## Lossless Zooming (4K to 1080p Zoom/Crop) On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-lossless-zooming#lossless-zooming))

[![Lossless Zooming](https://user-images.githubusercontent.com/18037362/144095838-d082040a-9716-4f8e-90e5-15bcb23115f9.gif)](https://youtu.be/8X0IcnkeIf8)

## Running Mask-RCNN On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-maskrcnn-resnet50#mask-r-cnn-on-depthai))

![Example](https://user-images.githubusercontent.com/56075061/145182204-af540962-f233-480c-82a0-56b2587e5072.gif)

## MegaDepth Neural Depth Running On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mega-depth#gen2-megadepth-on-depthai))

[MegaDepth](https://github.com/luxonis/depthai-experiments/blob/master/gen2-mega-depth/imgs/example.gif?raw=true)

## MJPEG Streaming From On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mjpeg-streaming#mjpeg-streaming-server))

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

## Class Agnostic Object Detector Running On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mobile-object-localizer#gen2-mobile-object-localizer-on-depthai))

![Image example](https://user-images.githubusercontent.com/18037362/140496684-e886fc00-612d-44dd-a6fe-c0d47988246f.gif)

## How to Use Multiple Cameras Simultaneously ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-multiple-devices#gen2-multiple-devices-per-host))

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")

## How to Sync NN Data with Image Data for Custom Neural Networks ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-nn-sync#gen2-nn-frame-sync))

![image](https://user-images.githubusercontent.com/5244214/104956823-36f31480-59cd-11eb-9568-64c0f0003dd0.gif)

## Optical Character Recognition in the Wild On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-ocr#how-to-run))

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749743-13febe00-5f01-11eb-8b5f-dca801f5d125.png)](https://www.youtube.com/watch?v=Bv-p76A3YMk "Gen2 OCR Pipeline")

## Palm Detection On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-palm-detection#palm-detection))

![Palm Detection](https://github.com/luxonis/depthai-experiments/blob/master/gen2-palm-detection/images/palm_detection.gif?raw=true)

## Pedestrian Re-Identification ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification#pedestrian-reidentification))

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

## People Counting On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter#gen2-people-counting))

[![image](https://user-images.githubusercontent.com/18037362/119807472-11c26580-bedb-11eb-907a-196b8bb92f28.png)](
https://www.youtube.com/watch?v=_cAP-yHhUN4)

## People Direction-Tracker and Counter ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker#gen2-people-tracker))

![demo](https://user-images.githubusercontent.com/18037362/145656510-94e12444-7524-47f9-a036-7ed8ee78fd7a.gif)

## Playing an On-Camera Encoded Stream on the Host ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-play-encoded-stream#gen2-play-encoded-stream))

![Encoding demo](https://user-images.githubusercontent.com/59799831/132475640-6e9f8b7f-52f4-4f75-af81-86c7f6e45b94.gif)

## Recording and Playing Back Depth in RealSense -Compatible Format ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay#record-and-replay))

![depth gif](https://user-images.githubusercontent.com/18037362/141661982-f206ed61-b505-4b17-8673-211a4029754b.gif)

## Road Segmentation On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-road-segmentation#gen2-road-segmentation-on-depthai))

![Road Segmentation on DepthAI](https://user-images.githubusercontent.com/5244214/130064359-b9534b08-0783-4c86-979b-08cbcaff9341.gif)

## Roboflow Integration ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-roboflow-integration#oak--roboflow-demo))

https://user-images.githubusercontent.com/26127866/147658296-23be4621-d37a-4fd6-a169-3ea414ffa636.mp4

## Social Distancing Example ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-social-distancing#gen2-social-distancing))

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## Text Blurring On-Device ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-text-blur#gen2-text-blurring-on-depthai))

![Text Blurring](https://github.com/luxonis/depthai-experiments/blob/master/gen2-text-blur/imgs/example.gif?raw=true)

## Image Classification On-Device ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-tf-image-classification#gen2-tensorflow-image-classification-example))

![Pedestrian Re-Identification](https://user-images.githubusercontent.com/5244214/109003919-522a0180-76a8-11eb-948c-a74432c22be1.gif)

## Facial Key-point Triangulation On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation#gen2-triangulation---stereo-neural-inference-demo))

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## WebRTC Streaming Example ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-webrtc-streaming#gen2-webrtc-streaming-example))

[![Gen2 WebRTC](https://user-images.githubusercontent.com/5244214/121884542-58a1bf00-cd13-11eb-851d-dc45d541e385.gif)](https://youtu.be/8aeqGgO8LjY)

## YOLO V3 V4 V5 X and P On-Camera ([here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo))

![yolo-logo](https://user-images.githubusercontent.com/56075061/144863247-fa819d1d-28d6-498a-89a8-c3f94d9e9357.gif)

