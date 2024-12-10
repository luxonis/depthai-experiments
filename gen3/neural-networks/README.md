# Overview
This section provides examples to help you get started with **AI model inference** using **DepthAI**.
The examples utilize publicly available models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models) and can be ran on a **Luxonis device** of choice.
The inference is made either on camera inputs or custom media files (image or video) and the results are displayed in the browser.

The examples are organized into two categories:
- [Basic example](basic-example/): Generic inference pipeline running a **single model** with a **single-image input** and a **single-head output**;
- [Advanced examples](advanced-examples/): Custom single- or multiple-model inference pipelines, with input that can be a single image, multiple images, or other data, and can have multi-head outputs.

Below we list all the available **Gen3 examples** (with links to their **Gen2 counterparts**).

# Basic Examples

## Classification

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [efficientnet-lite](https://hub.luxonis.com/ai/models/fdacd30d-97f4-4c55-843f-8b7e872d8acb?view=page) | [gen2-efficientnet-classification](../../gen2/gen2-efficientnet-classification) | YES | YES | [🔗](LINK_TO_MEDIA) |
| [image-quality-assessment](https://hub.luxonis.com/ai/models/1c43753e-5e4d-4d32-b08a-584098290d72?view=page) | [gen2-image-quality-assessment](../../gen2/gen2-image-quality-assessment) | YES | NO | [🔗](LINK_TO_MEDIA) |
| *[paddle-text-recognition](https://hub.luxonis.com/ai/models/9ae12b58-3551-49b1-af22-721ba4bcf269?view=page) | | YES | YES | [🔗](LINK_TO_MEDIA) |
| **[emotion-recognition](https://hub.luxonis.com/ai/models/3cac7277-2474-4b36-a68e-89ac977366c3?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |

"*" model works better when coupled with a text detector (e.g. padel-text-detection).

"**" model works better when coupled with a face detector (e.g. yunet).

## Object Detection

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [mobilenet-ssd](https://hub.luxonis.com/ai/models/2da6e0a5-4785-488d-8cf5-c35f7ec1a1ed?view=page) |TBA | TBA | NO | [🔗](LINK_TO_MEDIA) |
| [yolov6-nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) | [gen2-efficientDet](../../gen2/gen2-efficientDet) | YES | YES | [🔗](LINK_TO_MEDIA) |
| [yolov6-large](https://hub.luxonis.com/ai/models/7937248a-c310-4765-97db-0850086f2dd9?view=page) | | NO | YES | [🔗](LINK_TO_MEDIA) |
| [yolov10-nano](https://hub.luxonis.com/ai/models/03153a9a-06f7-4ce9-b655-3762d21d0a8a?view=page) | [gen2-efficientDet](../../gen2/gen2-efficientDet) | YES | YES | [🔗](LINK_TO_MEDIA) |
| [qrdet](https://hub.luxonis.com/ai/models/d1183a0f-e9a0-4fa2-8437-f2f5b0181739?view=page) | [gen2-qr-code-scanner](../../gen2/gen2-qr-code-scanner) | YES | YES | [🔗](LINK_TO_MEDIA) |
| [yunet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [scrfd-face-detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [scrfd-person-detection](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page) | | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [mediapipe-palm-detection](https://hub.luxonis.com/ai/models/9531aba9-ef45-4ad3-ae03-808387d61bf3?view=page) | [gen2-palm-detection](../../gen2/gen2-palm-detection) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [license-plate-detection](https://hub.luxonis.com/ai/models/7ded2dab-25b4-4998-9462-cba2fcc6c5ef?view=page) | TBT | YES | YES | [🔗](LINK_TO_MEDIA) |
| [ppe-detection](https://hub.luxonis.com/ai/models/fd8699bf-3819-4134-9374-3735b9660d3c?view=page) | TBT | YES | YES | [🔗](LINK_TO_MEDIA) |
| [paddle-text-detection](https://hub.luxonis.com/ai/models/131d855c-60b1-4634-a14d-1269bb35dcd2?view=page) | TBT | TBA | YES | [🔗](LINK_TO_MEDIA) |


## Keypoint Detection

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [yunet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [scrfd-face-detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [mediapipe-face-landmarker](https://hub.luxonis.com/ai/models/4632304b-91cb-4fcb-b4cc-c8c414e13f56?view=page) | [gen2-facemesh](../../gen2/gen2-facemesh) | YES | YES | [🔗](LINK_TO_MEDIA) |
| *[mediapipe-hand-landmarker](https://hub.luxonis.com/ai/models/42815cca-deab-4860-b4a9-d44ebbe2988a?view=page) | | TBA | TBA | [🔗](LINK_TO_MEDIA) |
| [yolov8-nano-pose-estimation](https://hub.luxonis.com/ai/models/12acd8d7-25c0-4a07-9dff-ab8c5fcae7b1?view=page) | [gen2-human-pose](../../gen2/gen2-human-pose) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [yolov8-large-pose-estimation](https://hub.luxonis.com/ai/models/8be178a0-e643-4f1e-b925-06512e4e15c7?view=page) | [gen2-human-pose](../../gen2/gen2-human-pose) | NO | YES | [🔗](LINK_TO_MEDIA) |
| **[lite-hrnet](https://hub.luxonis.com/ai/models/c7c9e353-9f6d-43e1-9b45-8edeae82db70?view=page) | [gen2-human-pose](../../gen2/gen2-human-pose) | YES | YES | [🔗](LINK_TO_MEDIA) |
| ***[superanimal-landmarker](https://hub.luxonis.com/ai/models/894cf1a2-23fb-4c96-8944-a0d1be38a7c7?view=page) | | YES | YES | [🔗](LINK_TO_MEDIA) |

"*" model works better when coupled with a hand detector.

"**" model works better when coupled with a human detector.

"***" model works better when coupled with an animal detector.

## Segmentation
| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [yolov8-instance-segmentation-nano](https://hub.luxonis.com/ai/models/9c1ea8c4-7ab4-46d2-954b-de237c7b4a05?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [yolov8-instance-segmentation-large](https://hub.luxonis.com/ai/models/698b881d-2e98-45d0-bc72-1121d2eb2319?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [mediapipe-selfie-segmentation](https://hub.luxonis.com/ai/models/dc85210d-5483-4fe2-86aa-16ad5d57d2d1?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [deeplab-v3-plus](https://hub.luxonis.com/ai/models/1189a661-fd0a-44fd-bc9e-64b94d60cb49?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [fastsam-s](https://hub.luxonis.com/ai/models/4af2416c-2ba4-4c85-97d0-fd26f089fc69?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [fastsam-x](https://hub.luxonis.com/ai/models/e7d3a0cf-7c1f-4e72-8c0c-9e2fcf53ca24?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [ewasr](https://hub.luxonis.com/ai/models/48ca429e-134e-486e-8f71-a8788fb7b510?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [yolo-p](https://hub.luxonis.com/ai/models/0a22d194-d525-46e7-a785-a267b7958a39?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [pp-liteseg](https://hub.luxonis.com/ai/models/5963005b-eab3-4b68-a24c-45f3b95c6b9d?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |


## Monocular Depth Estimation

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [midas-v2-1](https://hub.luxonis.com/ai/models/be09b09e-053d-4330-a0fc-0c9d16aac007?view=page) | [gen2-depth-mbnv2](../../gen2/gen2-depth-mbnv2), [gen2-fast-depth](../../gen2/gen2-fast-depth), [gen2-mega-depth](../../gen2/gen2-mega-depth) | TBA | YES | [🔗](LINK_TO_MEDIA) |
| [depth-anything-v2](https://hub.luxonis.com/ai/models/c5bf9763-d29d-4b10-8642-fbd032236383?view=page) |  | NO | YES | [🔗](LINK_TO_MEDIA) |

## Image-to-Image Translation

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [zero-dce](https://hub.luxonis.com/ai/models/8eaae754-6195-4766-a39c-2d19a856a492?view=page) |  | TBA | TBA | [🔗](LINK_TO_MEDIA) |
| [rt-super-resolution](https://hub.luxonis.com/ai/models/536a03d9-4901-4a4d-ab7a-0e12c472c48e?view=page) |  | TBA | TBA | [🔗](LINK_TO_MEDIA) |
| [esrgan](https://hub.luxonis.com/ai/models/0180f69d-04e7-4511-9d36-30c488b017ee?view=page) |  | TBA | TBA | [🔗](LINK_TO_MEDIA) |
| [dncnn3](https://hub.luxonis.com/ai/models/89c61463-1074-4f31-907f-751a83a9643a?view=page) |  | TBA | TBA | [🔗](LINK_TO_MEDIA) |

## Regression
| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [age-gender-recognition](https://hub.luxonis.com/ai/models/20cb86d9-1a4b-49e8-91ac-30f4c0a69ce1?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [gaze-estimation-adas](https://hub.luxonis.com/ai/models/b174ff1b-740b-4016-b8d5-b9488dbdd657?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [head-pose-estimation](https://hub.luxonis.com/ai/models/068ac18a-de71-4a6e-9f0f-42776c0ef980?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [l2cs-net](https://hub.luxonis.com/ai/models/7051c9d2-78a4-420b-91a8-2d40ecf958dd?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |

## Line Detection
| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [m-lsd](https://hub.luxonis.com/ai/models/9e3e01d8-2303-4113-bf69-cb10ec56ad5b?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [m-lsd-tiny](https://hub.luxonis.com/ai/models/1d879fef-2c5a-46f4-9077-fa99e29f79d8?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) |
| [ultra-fast-lane-detection](https://hub.luxonis.com/ai/models/b15d067f-2cde-48a0-85bf-52e1174b1ac0?view=page) | TBT | TBT | TBT | [🔗](LINK_TO_MEDIA) 

# Advanced Examples

| Gen3 | Gen2 | RVC2 | RVC4 | Illustration |
|------|------|------|------|--------------|
| ... (Gen3Example) | ... (Gen2Example) | ... (YES/NO/TBA)  | ... (YES/NO/TBA)  | ... (LinkToMedia) |