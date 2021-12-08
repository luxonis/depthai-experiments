# Full FOV NN inferencing

This demo shows how you can run NN inferencing on full FOV frames. Color camera's sensor has `4032x3040`
resultion, which can only be obtained from `isp` and `still` outputs of [ColorCamera](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/) node. In order to run NN inference on a frame, the frame must be in `RGB` format and needs to be specific size. We use [ImageManip](https://docs.luxonis.com/projects/api/en/latest/components/nodes/image_manip/) node to convert `YUV420` (`isp`) to `RGB` and to resize the frame to `300x300` (required by MobileNet NN that we use).

## Demo

![Full FOV]()

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
