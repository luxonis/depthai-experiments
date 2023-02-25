import time

import blobconverter
import depthai as dai

# Change the IP to your MQTT broker!
MQTT_BROKER = "test.mosquitto.org"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "test_topic/detections"

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(300, 300)
camRgb.setFps(10)
camRgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
camRgb.preview.link(nn.input)

script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

nn.out.link(script.inputs['detections'])

script_text = f"""
import time

mqttc = Client()
node.warn('Connecting to MQTT broker...')
mqttc.connect("{MQTT_BROKER}", {MQTT_BROKER_PORT}, 60)
node.warn('Successfully connected to MQTT broker!')

mqttc.loop_start()
cnt = 0
total = 0
while True:
    dets = node.io['detections'].get()
    total+=len(dets.detections)
    cnt+=1
    if cnt > 20:
        avrg = str(total / 20)
        cnt = 0
        total = 0
        node.warn(avrg)
        (ok, id) = mqttc.publish("{MQTT_TOPIC}", avrg, qos=2)
"""

with open("paho-mqtt.py", "r") as f:
    paho_script = f.read()
    script.setScript(f"{paho_script}\n{script_text}")

with dai.Device(pipeline) as device:
    print('Connected to OAK')
    while not device.isClosed():
        time.sleep(1)
