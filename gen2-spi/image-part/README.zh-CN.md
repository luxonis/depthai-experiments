[英文文档](README.md)

## Gen2 SPI 演示: 发送部分消息

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

该示例说明了如何仅接收消息的一部分，例如图像的一小部分。

### 在DepthAI上:
在 main.py 中，我们只是设置了一个SPIOut节点，该节点将输出300x300的颜色预览流。

### 在ESP32上:
ESP32仅请求300x300预览的一部分，将其接收并弹出消息/帧。

### 运行示例的ESP32端:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd ../esp32-spi-message-demo/image_part/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### 运行示例的DepthAI端:
`python3 main.py`

