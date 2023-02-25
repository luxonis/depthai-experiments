[英文文档](README.md)

# RSTP 流

此示例使您可以通过RTSP流式传输帧

## 安装依赖

```
sudo apt-get install gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev
python3 -m pip install -r requirements.txt
```

## 用法

运行应用程序

```
python3 main.py
```

要查看流式传输的帧，请使用带有以下链接的RSTP客户端（例如，VLC网络流）

```
rtsp://localhost:8554/preview
```

