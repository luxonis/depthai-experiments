#!/usr/bin/env python3

import subprocess
import depthai as dai

pipeline = dai.Pipeline()

# Nodes
colorCam = pipeline.createColorCamera()
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
monoCam = pipeline.createMonoCamera()
monoCam2 = pipeline.createMonoCamera()
ve1 = pipeline.createVideoEncoder()
ve2 = pipeline.createVideoEncoder()
ve3 = pipeline.createVideoEncoder()

ve1Out = pipeline.createXLinkOut()
ve2Out = pipeline.createXLinkOut()
ve3Out = pipeline.createXLinkOut()

# Properties
monoCam.setCamId(1)
monoCam2.setCamId(2)
ve1Out.setStreamName('ve1Out')
ve2Out.setStreamName('ve2Out')
ve3Out.setStreamName('ve3Out')

#setting to 26fps will trigger error
ve1.setDefaultProfilePreset(1280, 720, 25, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(3840, 2160, 25, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(1280, 720, 25, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Link nodes
monoCam.out.link(ve1.input)
colorCam.video.link(ve2.input)
monoCam2.out.link(ve3.input)

ve1.bitstream.link(ve1Out.input)
ve2.bitstream.link(ve2Out.input)
ve3.bitstream.link(ve3Out.input)

# Connect to the device
dev = dai.Device(pipeline)

# Prepare data queues
outQ1 = dev.getOutputQueue('ve1Out')
outQ2 = dev.getOutputQueue('ve2Out')
outQ3 = dev.getOutputQueue('ve3Out')

# Start the pipeline
dev.startPipeline()

# Processing loop
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
            break

print("Converting stream file (.h264/.h265) into a video file (.mp4)...")
cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
subprocess.check_call(cmd.format("mono1.h264", "mono1.mp4").split())
subprocess.check_call(cmd.format("mono2.h264", "mono2.mp4").split())
subprocess.check_call(cmd.format("color.h265", "color.mp4").split())
print("Conversion successful, check mono1.mp4 / mono2.mp4 / color.mp4")
