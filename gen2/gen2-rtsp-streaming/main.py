#!/usr/bin/env python3

import subprocess
import depthai as dai

ffmpeg_cmd = [
    'ffmpeg',
    '-re',  # Read input at native frame rate
    '-f', 'h264',  # Input format is H264
    '-fflags', 'nobuffer',  # Disable buffering
    '-flags', 'low_delay',  # Low latency mode
    '-i', '-',  # Read input from stdin (your H264 stream)
    '-c', 'copy',  # Copy the codec without re-encoding
    '-f', 'rtsp',  # Output format is RTSP
    'rtsp://localhost:8554/mystream'  # MediaMTX RTSP URL and path
]

# Open a subprocess to run the FFmpeg command
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

pipeline = dai.Pipeline()

FPS = 30
colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setFps(FPS)

videnc = pipeline.create(dai.node.VideoEncoder)
videnc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H264_MAIN)
colorCam.video.link(videnc.input)

veOut = pipeline.create(dai.node.XLinkOut)
veOut.setStreamName("encoded")
videnc.bitstream.link(veOut.input)


with dai.Device(pipeline) as device:
    encoded = device.getOutputQueue("encoded", maxSize=2, blocking=False)
    while device.isPipelineRunning():
        data = encoded.get().getData()
        process.stdin.write(data)
        process.stdin.flush()

# Close the stdin
process.stdin.close()
process.wait()