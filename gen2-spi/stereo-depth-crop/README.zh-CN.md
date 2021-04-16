[英文文档](README.md)

## Gen2 双目/裁剪 SPI 演示

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

### 在DepthAI上:
本示例说明如何裁剪StereoDepth节点的输出并将该数据长时间传递给ESP32。它的核心是gen2-camera-demo的简化版本。它将创建一个管道，该管道将向StereoDepth节点提供来自板载左右单目摄像头的输入，然后将StereoDepth节点的深度输出传递到SPI输出节点。SPIOut节点将使用自定义SPI协议将数据输出到ESP32。

### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。该协议隐藏在components / depthai-spi-api中的简单API后面。在此示例中，深度输出仍可能大于可用的空闲内存，因此我们还演示了SPI API中的回调，以逐包获取输出。请参阅./esp32-spi-message-demo/main/app_main.cpp了解ESP32的源代码，以更好地了解它的工作方式。

### 运行示例的ESP32端:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd ../esp32-spi-message-demo/jpeg_demo
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### 运行示例的DepthAI端:
`python3 main.py` - 在没有点云可视化的情况下运行

`python3 main.py -pcl` - 启用点云可视化

