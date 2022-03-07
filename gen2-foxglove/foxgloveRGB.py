import asyncio
import base64
import json
import math
import time
import cv2
import depthai as dai
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId


# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)


async def main():
    class Listener(FoxgloveServerListener):
        def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("First client subscribed to", channel_id)

        def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("Last client unsubscribed from", channel_id)

    async with FoxgloveServer("0.0.0.0", 8765, "example server") as server:
        server.set_listener(Listener())

        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        chan_id = await server.add_channel(
            {
                "topic": "image",
                "encoding": "json",
                "schemaName": "ros.sensor_msgs.CompressedImage",
                "schema": json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "header": {
                                "type": "object",
                                "properties": {
                                    "stamp": {
                                        "type": "object",
                                        "properties": {
                                            "sec": {"type": "integer"},
                                            "nsec": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                            "format": {"type": "string"},
                            "data": {"type": "string", "contentEncoding": "base64"},
                        },
                    },
                ),
            }
        )

        with dai.Device(pipeline) as device:

            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            while True:
                inRgb = qRgb.get()
                img = inRgb.getCvFrame()

                # get image from camera and encode it into .jpg format
                is_success, im_buf_arr = cv2.imencode(".jpg", img)

                # read from .jpeg format to buffer of bytes
                byte_im = im_buf_arr.tobytes()

                # data must be encoded in base64
                data = base64.b64encode(byte_im).decode("ascii")

                tmpTime = time.time_ns()
                sec = math.trunc(tmpTime / 1e9)
                nsec = tmpTime - sec

                # messages will be sent every 0.1 s
                await asyncio.sleep(0.1)

                # data is sent with json (data must be in above schema order)
                await server.send_message(
                    chan_id,
                    time.time_ns(),
                    json.dumps(
                        {
                            "header": {"stamp": {"sec": sec, "nsec": nsec}},
                            "format": "jpeg",
                            "data": data,
                        }
                    ).encode("utf8"),
                )


if __name__ == "__main__":
    run_cancellable(main())
