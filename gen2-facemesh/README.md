## [Gen2] Facial Landmarks on DepthAI

This example shows an implementation of Facial Landmark detection that's used in [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html). The TFLite model was taken from [MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark) and converted to blob so that it can run on OAK devices.

Model takes input image of size 192 x 192, and predicts 468 facial landmarks and a score. Note that this demo only supports once face. For multiface support, additional face detection stage must be first implemented.

We use input image 416 x 416 and resize it to 192 x 192 using ImageManip node.

![Image example](imgs/example.gif)

The model also includes effect renderer from 2D PNG image.

![Rendered effect](imgs/example_renderer.gif)

## Pre-requisites

1. Purchase a DepthAI (or OAK) model (see [shop.luxonis.com](https://shop.luxonis.com/)).

3. Install requirements.
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
python3 main.py [options]
```

Options:

* `-conf` or `--confidence_thresh`: Set score threshold. If score falls bellow the threshold, facial landmarks are not shown.
