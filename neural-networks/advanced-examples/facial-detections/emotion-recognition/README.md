# Emotion recognition
This example demonstrates how to run 2 stage inference with the DepthAI v3 library.
It recognizes all people in the frame and determines their emotions. There are 8 possible emotions: anger, contempt, disgust, fear, happiness, neutral, sadness, and surprise. The demo uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) face detection model, crops the faces and then recognizes the emotions of the person using [Emotions recognition model](https://hub.luxonis.com/ai/models/3cac7277-2474-4b36-a68e-89ac977366c3?view=page).

**:exclamation: This experiment works only on RVC4 devices:exclamation:**

## Demo

![Demo](visualizations/output.gif)

# Instalation
Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).You need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:
```bash
pip install -r requirements.txt
```
# Usage
You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --media <MEDIA> --fps_limit <FPS_LIMIT>
```

- `<DEVICE>`: Device IP or ID. Default: ``.
- `<MEDIA>`: Path to the video file. Default `None` - camera input.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

The relevant arguments:
- **--device** [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- **--media** [OPTIONAL]: Path to the media file to be used as input. 
Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.


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