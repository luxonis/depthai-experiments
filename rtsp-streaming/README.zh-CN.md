[英文文档](README.md)

# RSTP 流

此示例使您可以通过RTSP流式传输帧

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

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

