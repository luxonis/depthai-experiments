# Overview
This section provides examples to help you get started with **AI model inference** using **DepthAI**.
The examples utilize publicly available models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models) and can be ran on a **Luxonis device** of choice.
The inference is made either on camera inputs or custom media files (image or video) and the results are displayed in the browser.

The examples are organized into two categories:
- [Basic example](basic-example/): Generic inference pipeline running a single model with a single-image input;
- [Advanced examples](advanced-examples/): Custom single- or multiple-model inference pipelines, with input that can be a single image, multiple images, or other data.

Below we list all the available **Gen3 examples** (with links to their **Gen2 counterparts**).

# Basic Examples

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| ... (ModelName) | ... (Gen2Example) | ... (YES/NO/TBA)  | ... (YES/NO/TBA)  | ... (LinkToMedia) |

## Object Detection

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [yolov6-nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) | [gen2-efficientDet](../../gen2/gen2-efficientDet) | YES | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [yolov6-large](https://hub.luxonis.com/ai/models/7937248a-c310-4765-97db-0850086f2dd9?view=page) | | NO | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [yolov10-nano](https://hub.luxonis.com/ai/models/03153a9a-06f7-4ce9-b655-3762d21d0a8a?view=page) | [gen2-efficientDet](../../gen2/gen2-efficientDet) | YES | YES | [🔗](basic-example/illustrations/dummy.gif) |


## Keypoint Detection

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [yunet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [scrfd-face-detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) | [gen2-face-detection](../../gen2/gen2-face-detection) | TBA | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [mediapipe-face-landmarker](https://hub.luxonis.com/ai/models/4632304b-91cb-4fcb-b4cc-c8c414e13f56?view=page) | [gen2-facemesh](../../gen2/gen2-facemesh) | YES | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [yolov8-nano-pose-estimation](https://hub.luxonis.com/ai/models/12acd8d7-25c0-4a07-9dff-ab8c5fcae7b1?view=page) | [gen2-human-pose](../../gen2/gen2-human-pose) | TBA | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [yolov8-large-pose-estimation](https://hub.luxonis.com/ai/models/8be178a0-e643-4f1e-b925-06512e4e15c7?view=page) | [gen2-human-pose](../../gen2/gen2-human-pose) | NO | YES | [🔗](basic-example/illustrations/dummy.gif) |

## Monocular Depth Estimation

| HubAI Model | Gen2 | RVC2 | RVC4 | Illustration |
|-------------|------|------|------|--------------|
| [midas-v2-1](https://hub.luxonis.com/ai/models/be09b09e-053d-4330-a0fc-0c9d16aac007?view=page) | [gen2-depth-mbnv2](../../gen2/gen2-depth-mbnv2), [gen2-fast-depth](../../gen2/gen2-fast-depth), [gen2-mega-depth](../../gen2/gen2-mega-depth) | TBA | YES | [🔗](basic-example/illustrations/dummy.gif) |
| [depth-anything-v2](https://hub.luxonis.com/ai/models/c5bf9763-d29d-4b10-8642-fbd032236383?view=page) |  | NO | YES | [🔗](basic-example/illustrations/dummy.gif) |

# Advanced Examples

| Gen3 | Gen2 | RVC2 | RVC4 | Illustration |
|------|------|------|------|--------------|
| ... (Gen3Example) | ... (Gen2Example) | ... (YES/NO/TBA)  | ... (YES/NO/TBA)  | ... (LinkToMedia) |