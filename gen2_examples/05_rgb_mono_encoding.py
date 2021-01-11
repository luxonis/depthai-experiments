#!/usr/bin/env python3

import subprocess
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color and mono cameras
colorCam = pipeline.createColorCamera()
monoCam = pipeline.createMonoCamera()
monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoCam2 = pipeline.createMonoCamera()
monoCam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1 = pipeline.createVideoEncoder()
ve1.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
monoCam.out.link(ve1.input)

ve2 = pipeline.createVideoEncoder()
ve2.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
colorCam.video.link(ve2.input)

ve3 = pipeline.createVideoEncoder()
ve3.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
monoCam2.out.link(ve3.input)

# Create outputs
ve1Out = pipeline.createXLinkOut()
ve1Out.setStreamName('ve1Out')
ve1.bitstream.link(ve1Out.input)

ve2Out = pipeline.createXLinkOut()
ve2Out.setStreamName('ve2Out')
ve2.bitstream.link(ve2Out.input)

ve3Out = pipeline.createXLinkOut()
ve3Out.setStreamName('ve3Out')
ve3.bitstream.link(ve3Out.input)


# Pipeline defined, now the device is assigned and pipeline is started
dev = dai.Device(pipeline)
dev.startPipeline()

# Output queues will be used to get the encoded data from the outputs defined above
outQ1 = dev.getOutputQueue(name='ve1Out', maxSize=30, blocking=True)
outQ2 = dev.getOutputQueue(name='ve2Out', maxSize=30, blocking=True)
outQ3 = dev.getOutputQueue(name='ve3Out', maxSize=30, blocking=True)

# The .h264 / .h265 files are raw stream files (not playable yet)
with open('mono1.h264', 'wb') as file_mono1_h264, open('color.h265', 'wb') as file_color_h265, open('mono2.h264', 'wb') as file_mono2_h264:
    print("Press Ctrl+C to stop encoding...")
    while True:
        try:
            # Empty each queue
            while outQ1.has():
                outQ1.get().getData().tofile(file_mono1_h264)

            while outQ2.has():
                outQ2.get().getData().tofile(file_color_h265)

            while outQ3.has():
                outQ3.get().getData().tofile(file_mono2_h264)
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            break

print("Converting stream file (.h264/.h265) into a video file (.mp4)...")
# ffmpeg is used to convert a raw stream file to the playable .mp4 one
cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
subprocess.check_call(cmd.format("mono1.h264", "mono1.mp4").split())
subprocess.check_call(cmd.format("mono2.h264", "mono2.mp4").split())
subprocess.check_call(cmd.format("color.h265", "color.mp4").split())
print("Conversion successful, check mono1.mp4 / mono2.mp4 / color.mp4")
