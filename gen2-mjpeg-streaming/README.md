[中文文档](README.zh-CN.md)

# MJPEG Streaming server

This script allows you to:
- Stream frames via HTTP Server using MJPEG stream
- Stream data via TCP Server

## Demo

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")


## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

To see the streamed frames, open [localhost:8090](http://localhost:8090).  This works in Chrome, but not Firefox or Safari.

To see the streamed data, use

```
nc localhost 8070
```
