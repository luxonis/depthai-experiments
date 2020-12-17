## Gen2 JPEG/Large File SPI Demo

### Overview:
This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

### On the DepthAI:
In main.py, a basic pipeline consisting of just 3 nodes is built and sent over to the DepthAI. This pipeline takes the output from the onboard color camera, encodes it into a jpeg and then sends that jpeg out the SPI interface.

### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in components/depthai-spi-api. In this example, the jpeg images can still often be larger than available free memory so we also demostrate a callback in the SPI API to get the jpegs packet by packet. Please see ./esp32-spi-message-demo/main/app_main.cpp for the ESP32 side source to get a better idea of what it's doing.

### Install Dependencies and get Submodules:
`python3 -m pip install -r requirements.txt`
git submodule update --init --recursive

### Run the ESP32 Side of the Example:
# If you haven’t already, set up the ESP32 programmer
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### Run the DepthAI Side of the Example:
`python3 main.py`

