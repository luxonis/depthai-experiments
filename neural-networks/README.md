# Overview
This section provides examples to help you get started with **AI model inference** using **DepthAI**.
The examples utilize publicly available models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models) and can be ran on a **Luxonis device** of choice.
The inference is made either on camera inputs or custom media files (image or video) and the results are displayed in the browser.

The examples are organized into two categories:
- [Generic example](generic-example/): Generic inference pipeline running a **single model** with a **single-image input** and a **single-head output**;
- [Advanced examples](advanced-examples/): Custom single- or multiple-model inference pipelines, with inputs that can be a single image, multiple images, or other data, and can have multi-head output.

Below we list all the available **Gen3 examples** (with links to their **Gen2 counterparts**). The generic-examples can be ran using a simple script (see the [README](generic-example/README.md)). The advanced-examples are ran individually (click on the relevant gen3 link for more information).
# Examples
LEGEND: ✅: available; ❌: not available; 🚧: work in progress

### Classification

| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [efficientnet-lite](https://hub.luxonis.com/ai/models/fdacd30d-97f4-4c55-843f-8b7e872d8acb?view=page) | [gen2-efficientnet-classification](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientnet-classification) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [image-quality-assessment](https://hub.luxonis.com/ai/models/1c43753e-5e4d-4d32-b08a-584098290d72?view=page) | [gen2-image-quality-assessment](https://github.com/luxonis/depthai-experiments/tree/master/gen2-image-quality-assessment) | [generic-example](generic-example/) | ✅ | ❌ | 🚧 |  |
| [paddle-text-recognition](https://hub.luxonis.com/ai/models/9ae12b58-3551-49b1-af22-721ba4bcf269?view=page) | [gen2-seven-segment-recognition](https://github.com/luxonis/depthai-experiments/tree/master/gen2-seven-segment-recognition) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 | must be coupled with a text detection model (e.g. padel-text-detection) for optimal performance|
| [emotion-recognition](https://hub.luxonis.com/ai/models/3cac7277-2474-4b36-a68e-89ac977366c3?view=page) | [gen2-emotion-recognition](https://github.com/luxonis/depthai-experiments/tree/master/gen2-emotion-recognition) | [generic-example](generic-example/) | ✅ | 🚧 | 🚧 | must be coupled with a face detection model (e.g. yunet) for optimal performance |


### Object Detection

| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [mobilenet-ssd](https://hub.luxonis.com/ai/models/2da6e0a5-4785-488d-8cf5-c35f7ec1a1ed?view=page) | [gen2-efficientDet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientDet) | [generic-example](generic-example/) | ✅ | ❌ | 🚧 |  |
| [yolov6-nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) | [gen2-efficientDet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientDet) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolov6-large](https://hub.luxonis.com/ai/models/7937248a-c310-4765-97db-0850086f2dd9?view=page) | [gen2-efficientDet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientDet) | [generic-example](generic-example/) | ❌ | ✅ | 🚧 |  |
| [yolov10-nano](https://hub.luxonis.com/ai/models/03153a9a-06f7-4ce9-b655-3762d21d0a8a?view=page) | [gen2-efficientDet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-efficientDet) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [qrdet](https://hub.luxonis.com/ai/models/d1183a0f-e9a0-4fa2-8437-f2f5b0181739?view=page) | [gen2-qr-code-scanner](https://github.com/luxonis/depthai-experiments/tree/master/gen2-qr-code-scanner) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yunet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) | [gen2-face-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [scrfd-face-detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) | [gen2-face-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [scrfd-person-detection](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page) | | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [mediapipe-palm-detection](https://hub.luxonis.com/ai/models/9531aba9-ef45-4ad3-ae03-808387d61bf3?view=page) | [gen2-palm-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-palm-detection) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [license-plate-detection](https://hub.luxonis.com/ai/models/7ded2dab-25b4-4998-9462-cba2fcc6c5ef?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [ppe-detection](https://hub.luxonis.com/ai/models/fd8699bf-3819-4134-9374-3735b9660d3c?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [paddle-text-detection](https://hub.luxonis.com/ai/models/131d855c-60b1-4634-a14d-1269bb35dcd2?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolo-world-l](https://hub.luxonis.com/ai/models/6684e96f-11fc-4d92-8657-12a5fd8e532a?view=page) |  | 🚧 | 🚧 | 🚧 | 🚧 | multi-input |

### Keypoint Detection

| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [yunet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) | [gen2-face-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [scrfd-face-detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) | [gen2-face-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [mediapipe-face-landmarker](https://hub.luxonis.com/ai/models/4632304b-91cb-4fcb-b4cc-c8c414e13f56?view=page) | [gen2-facemesh](https://github.com/luxonis/depthai-experiments/tree/master/gen2-facemesh) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolov8-nano-pose-estimation](https://hub.luxonis.com/ai/models/12acd8d7-25c0-4a07-9dff-ab8c5fcae7b1?view=page) | [gen2-human-pose](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolov8-large-pose-estimation](https://hub.luxonis.com/ai/models/8be178a0-e643-4f1e-b925-06512e4e15c7?view=page) | [gen2-human-pose](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose) | [generic-example](generic-example/) | ❌ | ✅ | 🚧 |  |
| [lite-hrnet](https://hub.luxonis.com/ai/models/c7c9e353-9f6d-43e1-9b45-8edeae82db70?view=page) | [gen2-human-pose](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 | must be coupled with a human detection model for optimal performance |
| [superanimal-landmarker](https://hub.luxonis.com/ai/models/894cf1a2-23fb-4c96-8944-a0d1be38a7c7?view=page) | | [generic-example](generic-example/) | ✅ | ✅ | 🚧 | must be coupled with an animal detection model for optimal performance |
| [objectron](https://hub.luxonis.com/ai/models/4c7a51db-9cbe-4aee-a4c1-b8abbbe18c11?view=page) | | 🚧 | 🚧 | 🚧 | 🚧 | multi-head |
| [mediapipe-hand-landmarker](https://hub.luxonis.com/ai/models/42815cca-deab-4860-b4a9-d44ebbe2988a?view=page) | | 🚧 | 🚧 | 🚧 | 🚧 | multi-head; must be coupled with a hand detection model for optimal performance |

### Segmentation
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [yolov8-instance-segmentation-nano](https://hub.luxonis.com/ai/models/9c1ea8c4-7ab4-46d2-954b-de237c7b4a05?view=page) | [gen2-maskrcnn-resnet50](https://github.com/luxonis/depthai-experiments/tree/master/gen2-maskrcnn-resnet50), <br>[gen2-deeplabv3_multiclass](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolov8-instance-segmentation-large](https://hub.luxonis.com/ai/models/698b881d-2e98-45d0-bc72-1121d2eb2319?view=page) | [gen2-maskrcnn-resnet50](https://github.com/luxonis/depthai-experiments/tree/master/gen2-maskrcnn-resnet50), <br>[gen2-deeplabv3_multiclass](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [mediapipe-selfie-segmentation](https://hub.luxonis.com/ai/models/dc85210d-5483-4fe2-86aa-16ad5d57d2d1?view=page) | [gen2-deeplabv3_person](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_person) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [deeplab-v3-plus](https://hub.luxonis.com/ai/models/1189a661-fd0a-44fd-bc9e-64b94d60cb49?view=page) | [gen2-deeplabv3_multiclass](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass) | [generic-example](generic-example/) | ✅ | 🚧 | 🚧 |  |
| [fastsam-s](https://hub.luxonis.com/ai/models/4af2416c-2ba4-4c85-97d0-fd26f089fc69?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [fastsam-x](https://hub.luxonis.com/ai/models/e7d3a0cf-7c1f-4e72-8c0c-9e2fcf53ca24?view=page) |  | [generic-example](generic-example/) | ❌ | ✅ | 🚧 |  |
| [ewasr](https://hub.luxonis.com/ai/models/48ca429e-134e-486e-8f71-a8788fb7b510?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [pp-liteseg](https://hub.luxonis.com/ai/models/5963005b-eab3-4b68-a24c-45f3b95c6b9d?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [yolo-p](https://hub.luxonis.com/ai/models/0a22d194-d525-46e7-a785-a267b7958a39?view=page) | [gen2-road-segmentation](https://github.com/luxonis/depthai-experiments/tree/master/gen2-road-segmentation), <br>[gen2-lanenet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-lanenet) | 🚧 | 🚧 | 🚧 | 🚧 |  |

### Depth Estimation
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [crestereo](https://hub.luxonis.com/ai/models/4729a8bd-54df-467a-92ca-a8a5e70b52ab?view=page) |  | 🚧 | 🚧 | 🚧 | 🚧 | multi-input |
| [midas-v2-1](https://hub.luxonis.com/ai/models/be09b09e-053d-4330-a0fc-0c9d16aac007?view=page) | [gen2-depth-mbnv2](https://github.com/luxonis/depthai-experiments/tree/master/gen2-depth-mbnv2), <br>[gen2-fast-depth](https://github.com/luxonis/depthai-experiments/tree/master/gen2-fast-depth), <br>[gen2-mega-depth](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mega-depth) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [depth-anything-v2](https://hub.luxonis.com/ai/models/c5bf9763-d29d-4b10-8642-fbd032236383?view=page) |  | [generic-example](generic-example/) | ❌ | ✅ | 🚧 |  |

### Regression
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [age-gender-recognition](https://hub.luxonis.com/ai/models/20cb86d9-1a4b-49e8-91ac-30f4c0a69ce1?view=page) | [gen2-age-gender](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender) | 🚧 | 🚧 | 🚧 | 🚧 | multi-head; must be coupled with a face detection model (e.g. yunet) for optimal performance |
| [gaze-estimation-adas](https://hub.luxonis.com/ai/models/b174ff1b-740b-4016-b8d5-b9488dbdd657?view=page) | [gen2-gaze-estimation](https://github.com/luxonis/depthai-experiments/tree/master/gen2-gaze-estimation) | 🚧 | 🚧 | 🚧 | 🚧 | multi-input |
| [head-pose-estimation](https://hub.luxonis.com/ai/models/068ac18a-de71-4a6e-9f0f-42776c0ef980?view=page) | [gen2-head-posture-detection](https://github.com/luxonis/depthai-experiments/tree/master/gen2-head-posture-detection) | 🚧 | 🚧 | 🚧 | 🚧 | multi-head; must be coupled with a face detection model (e.g. yunet) for optimal performance |
| [l2cs-net](https://hub.luxonis.com/ai/models/7051c9d2-78a4-420b-91a8-2d40ecf958dd?view=page) |  | 🚧 | 🚧 | 🚧 | 🚧 | multi-head |

### Line Detection
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [m-lsd](https://hub.luxonis.com/ai/models/9e3e01d8-2303-4113-bf69-cb10ec56ad5b?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |
| [m-lsd-tiny](https://hub.luxonis.com/ai/models/1d879fef-2c5a-46f4-9077-fa99e29f79d8?view=page) |  | [generic-example](generic-example/) | ✅ | ❌ | 🚧 |  |
| [ultra-fast-lane-detection](https://hub.luxonis.com/ai/models/b15d067f-2cde-48a0-85bf-52e1174b1ac0?view=page) | [gen2-lanenet](https://github.com/luxonis/depthai-experiments/tree/master/gen2-lanenet) | [generic-example](generic-example/) | ✅ | ✅ | 🚧 |  |


### Image-to-Image Translation

| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [zero-dce](https://hub.luxonis.com/ai/models/8eaae754-6195-4766-a39c-2d19a856a492?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 | if providing media, make sure to convert it to grayscale |
| [rt-super-resolution](https://hub.luxonis.com/ai/models/536a03d9-4901-4a4d-ab7a-0e12c472c48e?view=page) |  | [generic-example](generic-example/) | ✅ | 🚧 | 🚧 |  |
| [esrgan](https://hub.luxonis.com/ai/models/0180f69d-04e7-4511-9d36-30c488b017ee?view=page) |  | [generic-example](generic-example/) | ❌ | ✅ | 🚧 | missing visualization |
| [dncnn3](https://hub.luxonis.com/ai/models/89c61463-1074-4f31-907f-751a83a9643a?view=page) |  | [generic-example](generic-example/) | ✅ | ✅ | 🚧 | missing visualization, grayscale input |


### Image Embedding
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [osnet](https://hub.luxonis.com/ai/models/6d853621-818b-4fa4-bd9a-d9bdcb5616e6?view=page) |  | [generic-example](generic-example/) | 🚧 | 🚧 | 🚧 | missing visualization |
| [arcface](https://hub.luxonis.com/ai/models/e24a577e-e2ff-4e4f-96b7-4afb63155eac?view=page) |  | [generic-example](generic-example/) | 🚧 | 🚧 | 🚧 | missing visualization |

### Feature Detection
| HubAI Model | Gen2 | Gen3 | RVC2 | RVC4 | Illustration | Notes |
|-------------|------|------|------|------|--------------|-------|
| [xfeat](https://hub.luxonis.com/ai/models/6c2790a1-bf68-4e89-a4b3-5c9ae68183b5?view=page) | ❌ | ✅ | ✅ | ✅ | [📸](advanced-examples/image-matching/xfeat/media/xfeat_demo.gif) |  |







