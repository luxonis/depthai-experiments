# Full FOV NN inferencing

See [Maximizing FOV documentation page](https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/) for more information.

This demo shows how you can run NN inferencing on full FOV frames. IMX378 sensor has `4032x3040`
resultion, which can only be obtained from `isp` and `still` outputs of [ColorCamera](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/) node. In order to run NN inference on a frame it must be of specific size, eg. 300x300, which is a different aspect ratio than the full FOV of the sensor.

Here we have 3 options:

1. Crop the ISP frame to 1:1 aspect ratio and lose some FOV - see [cropping.py](cropping.py) demo.
2. Stretch the ISP frame to to the 1:1 aspect ratio of the NN - see [stretching.py](stretching.py) demo.
3. Apply letterboxing to the ISP frame to get 1:1 aspect ratio frame - see [letterboxing.py](letterboxing.py) demo.

## Demos

### Cropping

![cropping](https://user-images.githubusercontent.com/18037362/180607873-6a476ea4-55e0-4557-a93e-a7cadcd80725.jpg)
### Letterboxing

![letterboxing](https://user-images.githubusercontent.com/18037362/180607958-0db7fb34-1221-42a1-b889-10d1f9793912.jpg)
### Stretching

![stretched](https://user-images.githubusercontent.com/18037362/180607962-e616cdc7-fcad-4bc8-a15f-617b89a2c047.jpg)


## ISP vs Video vs Preview at 12MP

![Isp vs Video vs Preview](https://user-images.githubusercontent.com/18037362/180610776-854c5215-8b59-4300-81d8-0014847a04bc.jpg)

Image above is the `isp` output frame from the `ColorCamera` (12MP from IMX378). Blue rectangle represents the cropped 4K
`video` output, and yellow rectangle represents cropped `preview` output when preview size is set to 1:1 aspect ratio
(eg. when using 300x300 MobileNet-SSD NN model). [Source code here](https://gist.github.com/Erol444/56e23ec203a122d540ebc4d01d894d44).

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 cropping.py
python3 letterboxing.py
python3 stretching.py
```
