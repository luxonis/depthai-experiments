# RTSP Streaming

This experiment demonstrates how to stream frames via RTSP server using H265 stream.

## Installation

### Ubuntu 20.04

```
sudo apt-get install ffmpeg gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-good gstreamer1.0-plugins-base
python3 -m pip install -r requirements.txt
```

### Mac OS 11 (Big Sur)

```
brew install pkg-config cairo gobject-introspection gst-plugins-bad gst-plugins-base gstreamer gst-rtsp-server ffmpeg gst-plugins-good
python3 -m pip install -r requirements.txt
```

(if you're using Mac ARM processor, you might have to configure your homebrew properly to install these packages - check [this StackOverflow question](https://stackoverflow.com/q/64882584))

### Windows

We'd suggest using WSL2 and installing Ubuntu on there. You can use the command apt-get command from Ubuntu above to install necessary dependencies.

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --fps_limit <FPS_LIMIT>
```

- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

#### Examples

```bash
python3 main.py
```

This will run the RTSP Streaming experiment with the default device and camera input.
To see the streamed frames, use a RTSP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/preview
```

On Ubuntu or Mac OS, you can use `ffplay` (part of `ffmpeg` library) to preview the stream

```
ffplay -fflags nobuffer -fflags discardcorrupt -flags low_delay -framedrop rtsp://localhost:8554/preview
```

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments
