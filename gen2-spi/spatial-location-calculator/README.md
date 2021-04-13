## Gen2 Depth calculator File SPI Demo

### Overview:
This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

This demo showcases device side result parsing for depth calculator node. The ROI (Region of interest) is preconfigured at pipeline build time. 

### On the DepthAI:
In main.py, a basic pipeline consisting of just 2 nodes is built and sent over to the DepthAI. 
1. The depth from the Stereo node is passed into a SpatialLocationCalculator node.
2. The SpatialLocationCalculator node calculates the average of the ROIs.
3. The SPIOut Node receives parsed results from SpatialLocationCalculator and sends them to the ESP32 via SPI.
4. Expected output: average depth of ROI + initial config for it (ROI, lower, upper threshold)


### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in components/depthai-spi-api. 

In this example the ESP32 receives parsed simplified results from depth calculator running on the DepthAI. The ESP32 simply decodes and prints the results in this example.

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/spatial_location_calculator/
idf.py build
idf.py -p /dev/ttyUSB* flash monitor
```

### Run the DepthAI Side of the Example:
From `gen2-spi/spatial_location_calculator`
`python3 main.py`
