# RTSP Streaming

This example allows you to stream frames using [MediaMTX](https://github.com/bluenviron/mediamtx), which is a media server and proxy that allows you to publish/subscribe to media streams.

## Installation

Install [MediaMTX](https://github.com/bluenviron/mediamtx), you can either go with [standalone binary release](https://github.com/bluenviron/mediamtx/releases) or [docker image](https://github.com/bluenviron/mediamtx?tab=readme-ov-file#docker-image). Note that at least on MacOS

## Usage

First, run the MediaMTX server:

```bash
$ ./mediamtx
2024/08/21 15:26:08 INF MediaMTX v1.8.5
2024/08/21 15:26:08 INF configuration loaded from /Users/erik/Downloads/mediamtx_v1.8.5_darwin_arm64/mediamtx.yml
2024/08/21 15:26:08 INF [RTSP] listener opened on :8554 (TCP), :8000 (UDP/RTP), :8001 (UDP/RTCP)
2024/08/21 15:26:08 INF [RTMP] listener opened on :1935
2024/08/21 15:26:08 INF [HLS] listener opened on :8888
2024/08/21 15:26:08 INF [WebRTC] listener opened on :8889 (HTTP), :8189 (ICE/UDP)
2024/08/21 15:26:08 INF [SRT] listener opened on :8890 (UDP)
```

Now let's run the main.py script, which will start publishing H264-encoded stream to the MediaMTX server.

```
python3 main.py
```

### View stream

To see the streamed frames, use a RTSP Client (e.g. VLC Network Stream) with the following link

```
rtsp://localhost:8554/preview
```

On Ubuntu or Mac OS, you can use `ffplay` (part of `ffmpeg` library) to preview the stream, which will provide better performance than VLC (400ms latency vs >1sec latency).

```
ffplay -fflags nobuffer -fflags discardcorrupt -flags low_delay -framedrop rtsp://localhost:8554/preview
```
