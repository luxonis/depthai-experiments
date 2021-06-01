# Gen2 SPI Examples
This directory contains a few examples of how to use the SPI interface with the Gen2 Pipeline builder. The ESP32 side of these examples can be found in the esp32-spi-message-demo submodule. The MyriadX side of the examples are in this directory and and described below.

### Install Dependencies and Init Submodules:
Note that all of these examples depend on the requirements specified in requirements.txt and may have git submodules.

```
python3 install_requirements.py
git submodule update --init --recursive
```
Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   

#### jpeg-transfer
This example builds a pipeline that returns jpegs of the color camera through SPI.

#### standalone-jpeg
This example shows you how to flash the gen2-spi-jpeg example to the board so it can be run standalone.

#### stereo-depth-crop
This example crops and returns a section of the depth map over SPI.

#### device-yolo-parsing
This example demonstrates how to use the DetectionNetwork to parse known types such as YOLO on the DepthAI rather than on the ESP32.

#### mobilenet-raw-parsing
This example shows how to pass back raw NeuralNetwork results over SPI. The example uses mobilenet results simply because it's results happen to be small and easy to parse.

#### esp32-spi-message-demo
This sub-repo contains the code that goes on an esp32, the receiver of the SPI data being sent from our MyriadX.

#### two-streams
This example shows how to send two different streams through the SPI to the ESP32.
