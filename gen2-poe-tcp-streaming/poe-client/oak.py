import depthai as dai
import time

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
camRgb.setIspScale(2,3)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEnc.input)

script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)
videoEnc.bitstream.link(script.inputs['frame'])

script.setScript("""
# Enter your own IP!
HOST_IP = "192.168.34.218"

import socket
import time

sock = socket.socket()
sock.connect((HOST_IP, 5000))

while True:
    pck = node.io["frame"].get()
    data = pck.getData()
    ts = pck.getTimestamp()
    header = f"ABCDE " + str(ts.total_seconds()).ljust(18) + str(len(data)).ljust(8)
    # node.warn(f'>{header}<')
    sock.send(bytes(header, encoding='ascii'))
    sock.send(data)
""")

with dai.Device(pipeline) as device:
    print("Connected")
    while True:
        time.sleep(1)
