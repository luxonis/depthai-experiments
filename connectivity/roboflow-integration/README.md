# Roboflow Demo
An app creating [Roboflow](https://roboflow.com) dataset using detections from an OAK camera.

## Demo
Live preview shows MobileNet SSD detections. After pressing `enter` the app grabs frames and uploads them to Roboflow dataset along with annotations.

https://user-images.githubusercontent.com/26127866/147658296-23be4621-d37a-4fd6-a169-3ea414ffa636.mp4

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
User controls
'enter' - to capture and upload frames with annotations.
'q' - quit
```

1. Setup Roboflow account
- Get API key ([app.roboflow.com](https://app.roboflow.com/) -> `settings` -> `workspaces` -> `Roboflow API` -> Copy private API key)
- Create new (empty) project at [app.roboflow.com](https://app.roboflow.com/). Then copy the workspace and project's (a.k.a. dataset's) name.

2. Run the application with your `API key`, `workspace name` and `dataset name`:

```
python3 main.py --dataset [DATASET_NAME] --workspace [WORKSPACE_NAME] --api-key [API_KEY]

arguments:
  required:
  -key [API_KEY], --api-key [API_KEY]
                        private API key copied from app.roboflow.com
  --workspace [WORKSPACE_NAME] 
                        name of the workspace in app.roboflow.com
  --dataset [DATASET_NAME]
                        name of the project in app.roboflow.com
  optional:
  -h, --help            show this help message and exit
  -ai [SECONDS], --autoupload-interval [SECONDS]
                        automatically upload annotations every [SECONDS] seconds (when used with --autoupload-threshold)
  -at [THRESHOLD], --autoupload-threshold [THRESHOLD]
                        automatically upload annotations with confidence above [THRESHOLD] (when used with --autoupload-interval)
  -res [WIDTHxHEIGHT], --upload-res [WIDTHxHEIGHT]
                        upload annotated images in [WIDTHxHEIGHT] resolution, which can be useful to create dataset with high-resolution images
```
