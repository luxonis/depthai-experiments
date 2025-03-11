# QR Code Detection with Tiling

This experiment uses [qrdet:nano-512x288](https://hub.luxonis.com/ai/models/d1183a0f-e9a0-4fa2-8437-f2f5b0181739?view=page) neural network to detect QR codes. These QR codes are then decoded on the host. The experiment utilizes tiling to divide the input frame into multiple smaller frames. Each smaller frame is passed to the QR detection network and processed independently.

## Demo

![example](media/example.gif)

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install the zbar library:

```
sudo apt-get install libzbar0
```

Install required packages by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode). `STANDALONE` mode is only supported on RVC4.

All scripts accept the following arguments:

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

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

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

Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
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
