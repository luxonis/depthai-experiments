# Roboflow Integration

This experiment shows how to create a [Roboflow](https://roboflow.com) dataset using detections from a **Luxonis device**. It uses [YoloV6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model for object detection.

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install required packages by running:

```bash
pip install -r requirements.txt
```

Create a Roboflow account, workspace and dataset and get your API key from [Roboflow](https://app.roboflow.com/) -> `settings` -> `API Keys` -> Copy private API key.

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode). `STANDALONE` mode is only supported on RVC4. The experiment will show a preview of the camera input and will upload detections to Roboflow dataset when `space` key is pressed. You can also set `--auto-interval` to automatically upload detections every `<SECONDS>` seconds.

Here is a list of all available parameters:

```
  -d DEVICE, --device DEVICE
                        Optional name, DeviceID or IP of the camera to connect
                        to. (default: None)
  -fps FPS_LIMIT, --fps_limit FPS_LIMIT
                        FPS limit for the model runtime. (default: None)
  -media MEDIA_PATH, --media_path MEDIA_PATH
                        Path to the media file you aim to run the model on. If
                        not set, the model will run on the camera input. (default: None)
  -key API_KEY, --api-key API_KEY
                        private API key copied from app.roboflow.com (default: None)
  --workspace WORKSPACE
                        Name of the workspace in app.roboflow.com (default: None)
  --dataset DATASET     Name of the project in app.roboflow.com (default: None)
  --auto-interval AUTO_INTERVAL
                        Automatically upload annotations every [SECONDS] seconds (default: None)
  --auto-threshold AUTO_THRESHOLD
                        Automatically upload annotations with confidence above
                        [AUTO_THRESHOLD] (when used with --auto-interval) (default: 0.5)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py --api-key <API_KEY> --workspace <WORKSPACE_NAME> --dataset <DATASET_NAME>
```

This will run the Roboflow Integration experiment with the default device and camera input. It will upload detections only when `space` key is pressed.

```bash
python3 main.py --api-key <API_KEY> --workspace <WORKSPACE_NAME> --dataset <DATASET_NAME> --auto-interval <SECONDS>
```

This will run the Roboflow Integration experiment with the default device and camera input. It will automatically upload detections every `<SECONDS>` seconds or when `space` key is pressed.

### Standalone Mode

Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

Replace `<API_KEY>`, `<WORKSPACE_NAME>` and `<DATASET_NAME>` with your Roboflow API key, workspace and dataset names in the [oakapp.toml](oakapp.toml) file.

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```
