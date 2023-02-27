# OAK PoE MQTT publishing

This example shows how you perform MQTT publishing directly from OAK POE camera. This would also work in [standalone mode](https://docs.luxonis.com/projects/api/en/latest/tutorials/standalone_mode/). Credits to [bherbruck](https://github.com/bherbruck) for porting the [paho.mqtt.python](https://github.com/eclipse/paho.mqtt.python) library to a single python file (`paho-mqtt.py`) which can be run inside the [Script node](https://docs.luxonis.com/projects/api/en/latest/components/nodes/script/).

For this demo I have used a public MQTT broker https://test.mosquitto.org with port 1883 which is unencrypted and unauthenticated. I have used MQTTLens to subscribe to messages that OAK POE camera is publishing.

Note that publishing too fast (eg 10FPS) won't work with https://test.mosquitto.org. In this demo the OAK POE will just publish the average number of objects it detects every ~2 seconds.

## Demo

![demo](https://user-images.githubusercontent.com/18037362/213188479-64e9a5aa-babe-47a5-96df-94f444204ac4.png)

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```
