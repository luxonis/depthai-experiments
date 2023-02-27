# Save encoded video stream into mp4 container

This demo shows how you can stream encoded frames from the device to the host and save them directly into a video container. [depthai-python](https://github.com/luxonis/depthai-python) examples like [RGB Encoding](https://docs.luxonis.com/projects/api/en/latest/samples/VideoEncoder/rgb_encoding/#rgb-encoding) will save encoded stream into `.h264` or `.mjpeg` file and require that user runs a `ffmpeg` command to convert raw encoded stream into a video container. **Video is encoded on the device itself** before it's sent to the host computer.

This demo uses [PyAV](https://github.com/PyAV-Org/PyAV) library, which is just a binding library for the [FFmpeg](http://ffmpeg.org/) library.

This demo will use `H265` codec by default. Note that some video players (eg. Quicktime) might not supprot this codec. We suggest using [VLC](https://www.videolan.org/vlc/).

## Demo

![image](https://user-images.githubusercontent.com/18037362/166504853-68072d92-f3ed-4a08-a7ca-15d7b8e774a2.png)

As you can see, the `video.mp4` uses the codec of the stream being saved, so there's no decoding/encoding (or converting) happening on the host computer and **host CPU/GPU/RAM usage is minimal**.

### Matroska

Besides ffmpeg and `.mp4` video container (which is patent encumbered), you could also use the `mkvmerge`
(see [MKVToolNix](https://mkvtoolnix.download/doc/mkvmerge.html) for GUI usage) and `.mkv` video container to mux encoded stream into video file that is supported by all major video players (eg. [VLC](https://www.videolan.org/vlc/))

```
mkvmerge -o vid.mkv video.h265
```

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
# For DepthAI API
python3 main.py codec
# codec can be either 'h264', 'h265', or 'mjpeg'.

# For DepthAI SDK
python3 main.py
```
