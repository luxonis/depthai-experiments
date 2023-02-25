[中文文档](README.zh-CN.md)

# RTSP Streaming

This example allows you to stream frames via RTSP

## Installation

### Ubuntu 20.04

```
sudo apt-get install ffmpeg gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-good gstreamer1.0-plugins-base
```

### Mac OS 11 (Big Sur)

```
brew install pkg-config cairo gobject-introspection gst-plugins-bad gst-plugins-base gstreamer gst-rtsp-server ffmpeg gst-plugins-good
```

(if you're using M1 processor, you might have to configure your homebrew properly to install these packages - check [this StackOverflow question](https://stackoverflow.com/q/64882584))

## Usage
### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

To see the streamed frames, use a RTSP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/preview
```

On Ubuntu or Mac OS, you can use `ffplay` (part of `ffmpeg` library) to preview the stream

```
ffplay -fflags nobuffer -fflags discardcorrupt -flags low_delay -framedrop rtsp://localhost:8554/preview
```