# Play Encoded Stream

This experiment shows how you can play H264/H265/MJPEG encoded streams on the host computer. It showcases three different ways of playing the encoded stream.

### 1. main.py using Visualizer

This plays encoded stream by passing it directly to Visualizer. This is the most straightforward way of playing the encoded stream, however, it is currently not possible to play H265 encoded streams using Visualizer.

### 2. pyav.py using PyAv library

This decodes encoded stream using the PyAv library and then displays them using Visualizer. This supports H264, MJPEG and H265 encoded streams.

### 3. mjpeg.py with OpenCV decoding

This decodes encoded stream using OpenCV's [cv2.imdecode()](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e) function and displays them using Visualizer. This example supports only MJPEG encoding. Note that MJPEG compression isn't as great compared to H.264/H.265.

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install required packages by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode). `STANDALONE` mode is only supported on RVC4.

All scripts accept the following arguments:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect
                    to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If
                    not set, the model will run on the camera input.
                    (default: None)
-enc {mjpeg,h265,h264}, --encode {mjpeg,h265,h264}
                    Select encoding format. (default: h264)

```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the experiment with default argument values and will play the encoded stream on the host computer.

```bash
python3 pyav.py -fps 10
```

This will run the the experiment at 10 FPS and will use PyAv to decode the encoded stream.

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
