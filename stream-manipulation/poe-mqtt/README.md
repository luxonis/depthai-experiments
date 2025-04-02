# PoE MQTT publishing

This example demonstrates, how to publish MQTT messages directly from a PoE camera. Credits to [bherbruck](https://github.com/bherbruck) for porting the [paho.mqtt.python](https://github.com/eclipse/paho.mqtt.python) library to a single python file (`paho-mqtt.py`) which can be run inside the [Script node](https://docs.luxonis.com/projects/api/en/latest/components/nodes/script/).

This example publishes a message containing the average number of objects detected in the last 10 seconds. It uses [YoloV6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model for object detection.

By default, the messages are published to a public broker `test.mosquitto.org` on port `1883`. They are published to the topic `test_topic/detections`. This can be changed with available CLI arguments (see Usage section below).

## Demo
![mqtt-client](media/mqtt_client.gif)
For the demo https://testclient-cloud.mqtt.cool/ was used.

## Installation

Running this example requires a PoE capable **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install requirements by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-b BROKER, --broker BROKER
                    MQTT broker address. (default: test.mosquitto.org)
-p PORT, --port PORT  MQTT broker port. (default: 1883)
-t TOPIC, --topic TOPIC
                    MQTT topic to publish to. (default: test_topic/detections)
-u USERNAME, --username USERNAME
                    MQTT broker username. (default: )
-pw PASSWORD, --password PASSWORD
                    MQTT broker password. (default: )
```

### Peripheral Mode

Running in peripheral mode requires a host computer. Below are some examples of how to run the experiment.

#### Examples

```bash
python3 main.py
```

This will run the experiment with default argument values and publish messages to `test.mosquitto.org` on port `1883`.

```bash
python3 main.py \
    --broker localhost \
    --port 1883 \
    --topic my_topic/detections \
    --username my_username \
    --password my_password
```

This will publish messages to `localhost` on port `1883`. It will use `my_username` and `my_password` credentials to authenticate to the broker. It will publish messages to the topic `my_topic/detections`.

### Standalone Mode

Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.
