#!/usr/bin/env python3

import subprocess
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create an encoder, consuming the frames and encoding them using H.265 encoding
videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam.video.link(videoEncoder.input)

# Create output
videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h265')
videoEncoder.bitstream.link(videoOut.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the encoded data from the output defined above
q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

# The .h265 file is a raw stream file (not playable yet)
with open('video.h265','wb') as videoFile:
    print("Press Ctrl+C to stop encoding...")
    try:
        while True:
            h264Packet = q.get()  # blocking call, will wait until a new data has arrived
            h264Packet.getData().tofile(videoFile)  # appends the packet data to the opened file
    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass

print("Converting stream file (.h265) into a video file (.mp4)...")
# ffmpeg is used to convert a raw .h265 file to the playable .mp4 one
subprocess.check_call("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4".split())
print("Conversion successful, check video.mp4")
