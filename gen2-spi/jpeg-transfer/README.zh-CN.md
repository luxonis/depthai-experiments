[英文文档](README.md)

## Gen2 JPEG/大文件 SPI 演示

### 概述:
此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

### 在DepthAI上:
在 main.py 中，构建了一个仅包含3个节点的基本管道，并将其发送到DepthAI。该管道从板载彩色摄像机获取输出，将其编码为jpeg，然后将该jpeg发送到SPI接口。

### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。该协议隐藏在components / depthai-spi-api中的简单API后面。在此示例中，jpeg图像通常仍大于可用的空闲内存，因此我们还演示了SPI API中的回调，以逐包获取jpegs。请参阅 ./esp32-spi-message-demo/main/app_main.cpp了解ESP32的源代码 ，以更好地了解它的工作方式。

### 运行示例的ESP32端:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html


本示例实际上有两个ESP32固件选项。拳头是一个更简单的示例，它简单地接收和丢弃从DepthAI发送的jpeg。

```
cd ../esp32-spi-message-demo/jpeg_demo/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

第二个是基于IDF文件服务器示例的演示固件。原始IDF示例可在以下位置找到:
https://github.com/espressif/esp-idf/tree/master/examples/protocols/http_server/file_serving


```
cd ../esp32-spi-message-demo/jpeg_webserver_demo/
idf.py menuconfig
# go to `Example Configuration` ->
#    1. WIFI SSID: WIFI network to which your PC is also connected to.
#    2. WIFI Password: WIFI password
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

服务器的IP地址将在启动时在控制台上打印出来。如果您在浏览器（例如 http://192.168.43.130/ ）中导航，您将找到“获取框架！”。页面上的按钮。单击，会将下一个传入的jpeg图像作为“ frame.jpg”写入ESP32的文件系统。

### 运行示例的DepthAI端:
`python3 main.py`

