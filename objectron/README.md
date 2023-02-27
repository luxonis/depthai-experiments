# Objectron

[Objectron](https://google.github.io/mediapipe/solutions/objectron.html) performs 3D object detection. It predicts the keypoints of a 3D bounding box around an object in the image plane!

We use the two-stage architecture, so it is specialized for certain classes. This demo uses the chair model. However, other models pretrained by Mediapipe include bikes, books, bottles, cameras, cereal boxes, chairs, cups, laptops, and shoes.

## Demo

![chair](https://user-images.githubusercontent.com/60359299/153616321-83f9d7ae-100a-4317-afd4-72bb3d1011d4.gif)

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```