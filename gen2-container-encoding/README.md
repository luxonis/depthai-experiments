# Save encoded video stream into mp4 container

This demo shows how you can stream encoded frames from the device to the host and save them directly into a video container. [depthai-python](https://github.com/luxonis/depthai-python) examples like [RGB Encoding](https://docs.luxonis.com/projects/api/en/latest/samples/VideoEncoder/rgb_encoding/#rgb-encoding) will save encoded stream into `.h264` or `.mjpeg` file and require that user runs a `ffmpeg` command to convert raw encoded stream into a video container. **Video is encoded on the device itself** before it's sent to the host computer.

This demo uses [PyAV](https://github.com/PyAV-Org/PyAV) library, which is just a binding library for the [FFmpeg](http://ffmpeg.org/) library.

This demo will use `H265` codec by default.

## Install requirements

```
python3 -m pip install -r requirements.txt
```

## Usage

```
Usage: python3 main.py [codec]
Example: python3 main.py h264

codec can be either 'h264', 'h265', or 'mjpeg'.
```
