[中文文档](README.zh-CN.md)

# RTSP Streaming

This example allows you to stream frames via RTSP

## Installation

```
sudo apt-get install gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev gstreamer1.0-plugins-bad
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

To see the streamed frames, use a RTSP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/preview
```

On Ubuntu, you can use `ffplay` to preview the stream

```
ffplay rtsp://localhost:8554/preview
```