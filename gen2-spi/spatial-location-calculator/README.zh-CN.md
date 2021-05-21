[英文文档](README.md)

## Gen2深度计算器文件SPI演示

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

该演示展示了深度计算器节点的设备端结果解析。在管道构建时预先配置了ROI（感兴趣的区域）。

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

### 在DepthAI上:
在main.py中，构建了一个仅包含2个节点的基本管道，并将其发送到DepthAI。
1. 来自“立体”节点的深度将传递到SpatialLocationCalculator节点中。
2. SpatialLocationCalculator节点计算ROI的平均值。
3. SPIOut节点从SpatialLocationCalculator接收解析的结果，并通过SPI将其发送到ESP32。
4. 预期输出：ROI的平均深度+初始配置（ROI，较低，较高阈值）


### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。该协议隐藏在components/depthai-spi-api中的简单API后面。

在本示例中，ESP32从运行在DepthAI上的深度计算器接收解析后的简化结果。在本示例中，ESP32只是解码并打印结果。

### 在ESP32端运行示例:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/spatial_location_calculator/
idf.py build
idf.py -p /dev/ttyUSB* flash monitor
```

### 在DepthAI端运行示例:
从 `gen2-spi/spatial_location_calculator`

`python3 main.py`
