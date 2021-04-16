[中文文档](README.zh-CN.md)

## Gen2 SPI Demo: Sending Parts of Messages

### Overview:
This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

The example shows how to receive just part of an message, such as a small section of an image. 

### On the DepthAI:
In main.py, we just set up a SPIOut node that will output a 300x300 color preview stream.

### On the ESP32:
The ESP32 requests just a section of the 300x300 preview, receives it and pops the message/frame. 

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd ../esp32-spi-message-demo/image_part/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### Run the DepthAI Side of the Example:
`python3 main.py`

