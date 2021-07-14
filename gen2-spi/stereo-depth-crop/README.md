[中文文档](README.zh-CN.md)

## Gen2 Stereo/Crop SPI Demo

### Overview:
This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

### On the DepthAI:
This example shows how to crop the output of StereoDepth nodes and pass that data long to an ESP32. At its core, it is a simplified version of the gen2-camera-demo. It will create a pipeline that will feed a StereoDepth node the inputs from the onboard left and right mono cameras, then pass the depth output of the StereoDepth node to a SPI out node. The SPIOut node will output data to a ESP32 using a custom SPI protocol. 

### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in components/depthai-spi-api. In this example, the depth output can still be larger than available free memory so we also demostrate a callback in the SPI API to get the output packet by packet. Please see ./esp32-spi-message-demo/main/app_main.cpp for the ESP32 side source to get a better idea of what it's doing.

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd ../esp32-spi-message-demo/jpeg_demo
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### Run the DepthAI Side of the Example:
`python3 main.py` - Runs without point cloud visualization
`python3 main.py -pcl` - Enables point cloud visualization

