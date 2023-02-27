# Full FOV NN inferencing

See [Maximizing FOV documentation page](https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/) for more information.

This demo shows how you can run NN inferencing on full FOV frames. IMX378 sensor has `4032x3040`
resultion, which can only be obtained from `isp` and `still` outputs of [ColorCamera](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/) node. In order to run NN inference on a frame it must be of specific size, eg. 300x300, which is a different aspect ratio than the full FOV of the sensor.

Here we have 3 options:

1. Crop the ISP frame to 1:1 aspect ratio and lose some FOV - see [cropping.py](sdk/cropping.py) demo.
2. Stretch the ISP frame to to the 1:1 aspect ratio of the NN - see [stretching.py](sdk/stretching.py) demo.
3. Apply letterboxing to the ISP frame to get 1:1 aspect ratio frame - see [letterboxing.py](sdk/letterboxing.py) demo.

## Demos

### Cropping

![cropping](https://user-images.githubusercontent.com/18037362/180607873-6a476ea4-55e0-4557-a93e-a7cadcd80725.jpg)
### Letterboxing

![letterboxing](https://user-images.githubusercontent.com/18037362/180607958-0db7fb34-1221-42a1-b889-10d1f9793912.jpg)
### Stretching

![stretched](https://user-images.githubusercontent.com/18037362/180607962-e616cdc7-fcad-4bc8-a15f-617b89a2c047.jpg)

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
python3 $SCRIPT_NAME.py
```
