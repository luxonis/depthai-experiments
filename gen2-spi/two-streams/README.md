## Two streams Demo

### Overview:
This example shows how to send two different streams through the SPI to the ESP32.

This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

### Demo

[![depthai](https://user-images.githubusercontent.com/18037362/118379752-bdfb7680-b5d4-11eb-9a56-848c2fdb57e8.gif)](https://www.youtube.com/watch?v=Ctiem0mHbkQ)

### On the DepthAI:
In main.py, a pipeline that has NN and SpatialLocationCalculator (SLC) is created. Pipeline sends NN (mobilenet) and SLC output to both host and ESP32.

### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in `components/depthai-spi-api`. ESP32 code is located at `esp32-spi-message-demo/two-streams/`, which will read both the SLC and NN outputs and just print it, so be sure
to add `monitor` argument when flashing the program to ESP32.

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/two-streams/
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```