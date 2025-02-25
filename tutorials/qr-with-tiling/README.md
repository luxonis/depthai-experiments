# QR Code Detection with Tiling

This experiment uses [qrdet:nano-512x288](https://hub.luxonis.com/ai/models/d1183a0f-e9a0-4fa2-8437-f2f5b0181739?view=page) neural network to detect QR codes. These QR codes are then decoded on the host. The experiment utilizes tiling to divide the input frame into multiple smaller frames. Each smaller frame is passed to the QR detection network and processed independently.

## Demo

![example](media/example.gif)

## Installation

1. Install the zbar library:

```
sudo apt-get install libzbar0
```

2. Install the Python dependencies:

```
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input.
-r ROWS, --rows ROWS
                    Number of rows in the grid for dividing the output into smaller frames. (default: 2)
-c COLUMNS, --columns COLUMNS
                    Number of columns in the grid for dividing the output into smaller frames. (default: 2)
-is INPUT_SIZE, --input_size INPUT_SIZE
                    Input video stream resolution. {2160p, 1080p, 720p} (default: 1080p)
```

#### Examples

```bash
python3 main.py
```

This will run the QR Code Detection with Tiling experiment with the default device and camera input.

```bash
python3 main.py -fps 10
```

This will run the QR Code Detection with Tiling experiment with the default device at 10 FPS.

```bash
python3 main.py -media /path/to/media.mp4
```

This will run the QR Code Detection with Tiling experiment with the default device and the specified media file.

```bash
python3 main.py -r 3 -c 3
```

This will run the QR Code Detection with Tiling experiment with the default device and the specified grid size.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments
