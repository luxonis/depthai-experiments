[英文文档](README.md)

## ESP端原始数据解析演示

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。 实现此目的的最简单方法是持有BW1092板。 它具有已经连接到DepthAI的集成ESP32。

此演示展示了针对已知网络类型的设备端结果解析。 诸如YOLO之类的网络的原始结果很快变得超出ESP32的处理能力。 为了处理常见的网络类型，可以使用DetectionNetwork节点。 DetectionNetwork节点接收一个神经网络Blob，并以ESP32可以读取的简化格式输出对象检测结果。

### 在DepthAI上:
在main.py中，构建了一个仅包含3个节点的基本管道，并将其发送到DepthAI。
1. 来自ColorCamera节点的预览将传递到YoloDetectionNetwork节点中。
2. YoloDetectionNetwork节点运行YOLO并解析结果。
3. SPOut节点从YoloDetectionNetwork接收解析结果，然后通过SPI将其发送到ESP32。

### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。 该协议隐藏在components / depthai-spi-api中的简单API后面。

在本示例中，ESP32从运行在DepthAI上的YOLO接收解析后的简化结果。 在本示例中，ESP32只是解码并打印结果。

### 运行ESP32端的示例:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/parse_meta/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### 运行DepthAI端的示例:
`python3 main.py mobilenet-ssd.blob.sh8cmx8NCE1`
