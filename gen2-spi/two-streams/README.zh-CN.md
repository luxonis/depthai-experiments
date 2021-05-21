[英文文档](README.md)

## 两流演示

### 概述:

本示例说明如何通过SPI向ESP32发送两个不同的流。

此演示需要一个通过SPI连接到DepthAI的ESP32板。 实现此目的的最简单方法是持有BW1092板。 它具有已经连接到DepthAI的集成ESP32。

### DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

### 演示

[![depthai](https://user-images.githubusercontent.com/18037362/118379752-bdfb7680-b5d4-11eb-9a56-848c2fdb57e8.gif)](https://www.youtube.com/watch?v=Ctiem0mHbkQ)

### 在DepthAI上运行:
在main.py中，创建具有NN和SpatialLocationCalculator（SLC）的管道。 管道将NN（移动网）和SLC输出发送到主机和ESP32。

### 在ESP32上运行:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。 该协议隐藏在一个简单的API中，该API存在于 `components/depthai-spi-api`. ESP32代码位于 `esp32-spi-message-demo/two-streams/`, 它会读取SLC和NN输出并只打印它，因此请确保添加 `monitor` 将程序刷新到ESP32时的实参。

### 在ESP32端运行示例:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/two-streams/
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```