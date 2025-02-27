# Emotion recognition

This example demonstrates how to run 2 stage inference with the DepthAI v3 library.
It recognizes all people in the frame and determines their emotions. There are 8 possible emotions: anger, contempt, disgust, fear, happiness, neutral, sadness, and surprise. The demo uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) face detection model, crops the faces and then recognizes the emotions of the person using [Emotions recognition model](https://hub.luxonis.com/ai/models/3cac7277-2474-4b36-a68e-89ac977366c3?view=page).

**:exclamation: This experiment works only on RVC4 devices:exclamation:**

## Demo

![Demo](media/emotion-recognition.gif)

## Instalation

Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).You need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
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

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device. In this case, remember to connect to `http://device-ip:8082` instead of localhost. To run in standalone, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <device-ip>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.
