import asyncio
import base64
import json
import struct
import time
import math
import numpy as np
import open3d as o3d
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId


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
                "topic": "pointCloud",
                "encoding": "json",
                "schemaName": "ros.sensor_msgs.PointCloud2",
                "schema": json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "header": {
                                "type": "object",
                                "properties": {
                                    "seq": {"type": "integer"},
                                    "stamp": {
                                        "type": "object",
                                        "properties": {
                                            "sec": {"type": "integer"},
                                            "nsec": {"type": "integer"},
                                        },
                                    },
                                    "frame_id": {"type": "string"}
                                },
                            },
                            "height": {"type": "integer"},
                            "width": {"type": "integer"},
                            "fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "offset": {"type": "integer"},
                                        "datatype": {"type": "integer"},
                                        "count": {"type": "integer"}
                                    }
                                },
                            },
                            "is_bigendian": {"type": "boolean"},
                            "point_step": {"type": "integer"},
                            "row_step": {"type": "integer"},
                            "data": {"type": "string", "contentEncoding": "base64"},
                            "is_dense": {"type": "boolean"}
                        },
                    },
                ),
            }
        )
        seq = 0

        # read point cloud from file, for real time this will be the point cloud that the camera creates
        ball = o3d.io.read_point_cloud("sphere.ply")
        points = np.asarray(ball.points)

        # write points to a buffer
        buf = bytes()
        for point in points:
            buf += struct.pack('f', float(point[0]))
            buf += struct.pack('f', float(point[1]))
            buf += struct.pack('f', float(point[2]))

        # data needs to be encoded in base64
        data = base64.b64encode(buf).decode("ascii")

        while True:

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
                        "header": {
                            "seq": seq,
                            "stamp": {"sec": sec, "nsec": nsec},
                            "frame_id": "front"
                        },
                        "height": 1,
                        "width": len(ball),
                        "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                                   {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                                   {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
                        "is_bigendian": False,
                        "point_step": 12,
                        "row_step": 12 * len(ball),
                        "data": data,
                        "is_dense": True
                    }
                ).encode("utf8"),
            )
            seq += 1


if __name__ == "__main__":
    run_cancellable(main())
