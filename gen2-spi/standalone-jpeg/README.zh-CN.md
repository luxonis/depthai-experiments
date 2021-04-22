[英文文档](README.md)

## Flash Bootloader 演示

### 概述:
这是在DepthAI上刷新/更新引导加载程序，以及将流水线刷新到将在引导时运行的设备的示例。这个例子实际上会将gen2-spi-jpeg例子刷新到板上并在启动时运行。

此演示需要一个通过SPI连接到DepthAI的ESP32板。实现此目的的最简单方法是持有BW1092板。它具有已经连接到DepthAI的集成ESP32。

### 在DepthAI上:
在 main.py 中，构建了一个仅包含3个节点的基本管道，并将其发送到DepthAI。该管道从板载彩色摄像机获取输出，将其编码为jpeg，然后将该jpeg发送到SPI接口。

### 在ESP32上:
ESP32运行自定义协议，以通过SPI与DepthAI进行通信。该协议隐藏在components / depthai-spi-api中的简单API后面。在此示例中，jpeg图像通常仍大于可用的空闲内存，因此我们还演示了SPI API中的回调，以逐包获取jpegs。请参阅./esp32-spi-message-demo/main/app_main.cpp了解ESP32的源代码，以更好地了解它的工作方式。

### 运行示例的ESP32端:
如果尚未安装，请设置ESP32 IDF框架:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/jpeg_demo/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### 运行示例的DepthAI端:
#### 将GPIO引脚设置为启用独立启动模式
https://docs.google.com/document/d/1Q0Wwjs0djMQOPwRT04k8tL20WWv_5AdwiQcPSeebqsw/edit

独立启动引脚:
![image](https://user-images.githubusercontent.com/19913346/102914698-ee801f80-443d-11eb-96f4-5cc0a5bfb263.png)

XLink引导引脚:
![image](https://user-images.githubusercontent.com/19913346/102914744-ff309580-443d-11eb-9975-66a7f633da6a.png)

#### 刷新引导程序
此步骤实际上只需要执行一次。如有必要，可以再次运行它以更新引导加载程序。

`python3 main.py bootloader`

#### 刷新管道
`python3 main.py`

此时，您可以重启电路板，并观察SPI输出中是否有传入数据。
