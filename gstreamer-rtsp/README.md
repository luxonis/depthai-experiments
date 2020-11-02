In your code create a GstServer object and pass inference results and frame from DepthAI pipeline to the "send_frame" method. Then simply connect to the rtsp stream using a rtsp client (vlc network stream) rtsp://192.168.x.x:8554/preview

```
sudo apt-get install gir1.2-gst-rtsp-server-1.0 python3-gst-1.0 gstreamer-1.0 python3-gi
python3 rstp.py
```


