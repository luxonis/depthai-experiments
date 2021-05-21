[英文文档](README.md)

## Gen2设备端空间Mobilenet解析演示

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

此演示展示了针对已知网络类型的设备端结果解析。来自诸如Mobilenet-SSD之类的网络的原始结果很快变得超出ESP32的处理能力。为了处理常见的网络类型，可以使用DetectionNetwork节点。DetectionNetwork节点接收一个神经网络Blob，并以ESP32可以读取的简化格式输出对象检测结果。

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

### 在DepthAI上:
在main.py中，构建了一个基本管道并将其发送到DepthAI。
1. 来自ColorCamera节点的预览将传递到MobileNetSpatialDetectionNetwork节点。
2. MobileNetSpatialDetectionNetwork节点运行Mobilenet-SSD并解析结果。
3. SPIOut节点从MobileNetSpatialDetectionNetwork接收解析的结果，并通过SPI将其发送到ESP32。

### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。该协议隐藏在components / depthai-spi-api中的简单API后面。

在此示例中，ESP32从运行在DepthAI上的Mobilenet-SSD接收解析后的简化结果。在本示例中，ESP32只是解码并打印结果。

### 在ESP32端运行示例:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/spatial_image_detections/
idf.py build
idf.py -p /dev/ttyUSB* flash monitor
```

### 在DepthAI端运行示例:
From `gen2-spi/spatial-mobilenet`

`python3 main.py mobilenet-ssd_openvino_2021.2_6shave.blob`
