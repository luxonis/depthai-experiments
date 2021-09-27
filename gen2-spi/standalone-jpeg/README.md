[中文文档](README.zh-CN.md)

## Flash Bootloader Demo

### Overview:
This is an example of flashing/updating the bootloader on DepthAI as well as flashing a pipeline to the device that'll be run at boot. This example will essentially flash the gen2-spi-jpeg example to the board and run it at boot.

This demo requires an ESP32 board connected via SPI to the DepthAI. The easiest way to accomplish this is to get a hold of an BW1092 board. It has an integrated ESP32 already connected to the DepthAI.

### On the DepthAI:
In main.py, a basic pipeline consisting of just 3 nodes is built and sent over to the DepthAI. This pipeline takes the output from the onboard color camera, encodes it into a jpeg and then sends that jpeg out the SPI interface.

### On the ESP32:
The ESP32 is running a custom protocol to communicate over SPI with the DepthAI. This protocol is hidden behind a simple API that lives in components/depthai-spi-api. In this example, the jpeg images can still often be larger than available free memory so we also demostrate a callback in the SPI API to get the jpegs packet by packet. Please see ./esp32-spi-message-demo/main/app_main.cpp for the ESP32 side source to get a better idea of what it's doing.

### Run the ESP32 Side of the Example:
If you haven’t already, set up the ESP32 IDF framework:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

```
cd esp32-spi-message-demo/jpeg_demo/
idf.py build
idf.py -p /dev/ttyUSB1 flash
```

### Run the DepthAI Side of the Example:
#### Set the GPIO pins to Enable Standalone Boot Mode
https://docs.google.com/document/d/1Q0Wwjs0djMQOPwRT04k8tL20WWv_5AdwiQcPSeebqsw/edit

Standalone Boot Pins:
![image](https://user-images.githubusercontent.com/19913346/102914698-ee801f80-443d-11eb-96f4-5cc0a5bfb263.png)

XLink Boot Pins:
![image](https://user-images.githubusercontent.com/19913346/102914744-ff309580-443d-11eb-9975-66a7f633da6a.png)

#### Flash the Bootloader
This step really only needs to be done once. It can be run again to update the bootloader if necessary.

`python3 main.py bootloader`

#### Flash the Pipeline
`python3 main.py`

At this point, you can reboot the board and watch the SPI output for incoming data. 
