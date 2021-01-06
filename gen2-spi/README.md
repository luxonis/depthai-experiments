# Gen2 SPI Examples
This directory contains a few examples of how to use the SPI interface with the Gen2 Pipeline builder.

#### gen2-spi-jpeg
This example builds a pipeline that returns jpegs of the color camera through SPI.

#### gen2-spi-standalone
This example shows you how to flash the gen2-spi-jpeg example to the board so it can be run standalone.

#### gen2-spi-stereo
This example crops and returns a section of the depth map over SPI.

#### device-yolo-parsing
This example demonstrates how to use the DetectionNetwork to parse known types such as YOLO on the DepthAI rather than on the ESP32.

#### esp32-spi-message-demo
This sub-repo contains the code that goes on an esp32, the receiver of the SPI data being sent from our MyriadX.
