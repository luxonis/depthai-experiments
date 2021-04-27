[中文文档](README.zh-CN.md)

# RSTP Streaming

This example allows you to stream frames via RTSP

## Installation

```
sudo apt-get install gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

To see the streamed frames, use a RSTP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/preview
```

