# Fatigue Detections

This example demonstrates how to detect if a person is fatigued and tired. Firstly, face is detected using the [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) face detection model. It is then cropped and [MediaPipe](https://hub.luxonis.com/ai/models/4632304b-91cb-4fcb-b4cc-c8c414e13f56?view=page) face landmark recognition model is used to detectec all keypoints. These keypoints are then used to predict the if the person is leaning forward and if their eyes are closed. If this is true for a around one second, the example prints a warning on the screen.

![example](media/output.gif)

## Instalation

Running this example requires a **Luxonis OAK device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: 5 for RVC2 and 7 for RVC4)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the experiment with default arguments.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.
