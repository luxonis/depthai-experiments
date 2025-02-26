# Display detections on higher resolution frames

Object detection models usually require smaller frame for inferencing (eg. `512x288`). Instead of displaying bounding boxes on such small frames, you could also stream higher resolution frames and display bounding boxes on these frames. There are several approaches to achieving that, and this experiment will go over some of them. It uses [YoloV6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model for object detection.

### 1. Passthrough

Simplest approach is to just stream the small inferencing frame. This example uses `passthrough` frame of `ParsingNeuralNetwork`'s output so bounding boxes are in sync with the frame.

![passthrough](media/passthrough_example.png)

### 2. Crop high resolution frame

A simple solution to low resolution frame is to use higher resolution frames and crop them to the correct size of the NN input. This example crops `640x480` frame to `512x288`.

![crop_highres](media/crop_highres_example.png)

### 3. Stretch or crop the frame before inferencing but keep the high resolution frame

Another solution is to stretch the frame to the correct aspect ratio and size of the NN. For more information, see [Resolution Techniques for NNs](https://docs.luxonis.com/software/depthai/resolution-techniques). This example stretches `1920x1440` frame to `512x288` before inferencing.

![stretch_before_inferencing](media/stretch_before_inferencing_example.png)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

This experiment contains 3 scripts. One for each approach mentioned above.

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
```

#### Examples

```bash
python3 passthrough.py
```

This will run the Display Detections experiment with the default device and camera input and use passthrough frame.

```bash
python3 crop_highres.py
```

This will run the Display Detections experiment with the default device and camera input and crop `640x480` frame to `512x288`.

```bash
python3 stretch_before_inferencing.py -fps 10
```

This will run the Display Detections experiment with the default device at 10 FPS and stretch `1920x1440` frame to `512x288`.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments
