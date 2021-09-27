[中文文档](README.zh-CN.md)

## Gen2 JPEG/Large File SPI Demo

### Overview:
This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

### On the DepthAI:
In main.py, a basic pipeline consisting of just 3 nodes is built and sent over to the DepthAI. This pipeline takes the output from the onboard color camera, encodes it into a jpeg and then sends that jpeg out the SPI interface.

### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in components/depthai-spi-api. In this example, the jpeg images can still often be larger than available free memory so we also demostrate a callback in the SPI API to get the jpegs packet by packet. Please see ./esp32-spi-message-demo/main/app_main.cpp for the ESP32 side source to get a better idea of what it's doing.

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html


There are actually two options of ESP32 firmware for this example. Fist is a more minimalistic example that simply receives and discards the jpeg transmitted from the DepthAI.

```
cd ../esp32-spi-message-demo/jpeg_demo/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

The second is an demo firmware is based off the IDF file server example. The original IDF example can be found here:
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

The IP address of the server will be printed out on the console as it starts up. If you navigate there in a browser (eg http://192.168.43.130/) you'll find a "Get Frame!" button on the page. Clicking that will write the next incoming jpeg image to ESP32's file system as 'frame.jpg'.

### Run the DepthAI Side of the Example:
`python3 main.py`

