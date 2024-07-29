import paho.mqtt.client as mqtt

# Define the MQTT broker details
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "test_topic/detections"

# Define the callback function for when a message is received
def on_message(client, userdata, message):
	print(f"Received message: {message.payload.decode()} on topic {message.topic}")

# Create an MQTT client instance
client = mqtt.Client()

# Assign the on_message callback function
client.on_message = on_message

# Connect to the MQTT broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Subscribe to the topic
client.subscribe(MQTT_TOPIC)

# Start the MQTT client loop to process messages
client.loop_start()

# Keep the script running to listen for messages
try:
	while True:
		pass
except KeyboardInterrupt:
	print("Exiting...")
finally:
	client.loop_stop()
	client.disconnect()