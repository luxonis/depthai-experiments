import depthai as dai
import time

# Start defining a pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
camRgb.setIspScale(2,3)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEnc.input)

script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

videoEnc.bitstream.link(script.inputs['frame'])
script.inputs['frame'].setBlocking(False)
script.inputs['frame'].setQueueSize(1)

script.setScript("""
import socket
import time

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen()
node.warn("Server up")

while True:
    conn, client = server.accept()
    node.warn(f"Connected to client IP: {client}")
    try:
        while True:
            pck = node.io["frame"].get()
            data = pck.getData()
            ts = pck.getTimestamp()
            header = f"ABCDE " + str(ts.total_seconds()).ljust(18) + str(len(data)).ljust(8)
            # node.warn(f'>{header}<')
            conn.send(bytes(header, encoding='ascii'))
            conn.send(data)

    except Exception as e:
        node.warn("Client disconnected")
""")

with dai.Device(pipeline) as device:
    print("Connected")
    while True:
        time.sleep(1)
