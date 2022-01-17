# Full FOV NN inferencing

This demo shows how you can run NN inferencing on full FOV frames. Color camera's sensor has `4032x3040`
resultion, which can only be obtained from `isp` and `still` outputs of [ColorCamera](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/) node. In order to run NN inference on a frame, the frame must be in `RGB` format and needs to be specific size. We use [ImageManip](https://docs.luxonis.com/projects/api/en/latest/components/nodes/image_manip/) node to convert `YUV420` (`isp`) to `RGB` and to resize the frame to `300x300` (required by MobileNet NN that we use).


## Demo

As you can see on the top left side of the image, `isp` frame resolution is `812x608` (downscaled from `4032x3040` - Full FOV of the camera).

![Full FOV](https://user-images.githubusercontent.com/18037362/145134354-7e6b2459-e4d5-4160-bd4f-d29458d30dad.png)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
