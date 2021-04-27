[英文文档](README.md)

# Gen2 SPI 示例
该目录包含一些有关如何在Gen2 Pipeline构建器中使用SPI接口的示例。这些示例的ESP32端可以在esp32-spi-message-demo子模块中找到。示例的MyriadX端在此目录中，并在下面进行描述。

### 安装依赖项和初始化子模块:
请注意，所有这些示例均取决于requirements.txt中指定的要求，并且可能具有git子模块。

```
python3 install_requirements.py
git submodule update --init --recursive
```
注意: `python3 install_requirements.py` 还会尝试从requirements-optional.txt中安装可选的库。例如：它包含open3d lib，这对于点云可视化是必需的。但是，该库的二进制文件不适用于树莓派和jetson之类的某些主机。

#### jpeg-transfer
本示例构建了一个通过SPI返回彩色相机jpeg的管道。

#### standalone-jpeg
本示例说明如何将gen2-spi-jpeg示例刷新到板上，使其可以独立运行。

#### stereo-depth-crop
本示例裁剪并通过SPI返回深度图的一部分。

#### device-yolo-parsing
本示例演示了如何使用DetectionNetwork在DepthAI而不是ESP32上解析已知类型，例如YOLO。

#### mobilenet-raw-parsing
本示例说明了如何通过SPI传回原始的NeuralNetwork结果。该示例使用mobilenet结果的原因很简单，因为它的结果很小并且很容易解析。

#### esp32-spi-message-demo
此子仓库包含esp32上的代码，SPI数据的接收器从MyriadX发送。
