#!/usr/bin/env python3
import argparse
from pathlib import Path
from multiprocessing import Process, Queue
import cv2
import depthai as dai
from datetime import timedelta
import contextlib
import math
import time

def check_range(min_val, max_val):
    def check_fn(value):
        ivalue = int(value)
        if min_val <= ivalue <= max_val:
            return ivalue
        else:
            raise argparse.ArgumentTypeError(
                "{} is an invalid int value, must be in range {}..{}".format(value, min_val, max_val)
            )
    return check_fn

def convert_to_mp4(path, fps, deleteMjpeg = False):
    try:
        import ffmpy3
        path_output = path.parent / (path.stem + '.mp4')
        print(path_output)
        print(path)
        try:
            ff = ffmpy3.FFmpeg(
                inputs={str(path): "-y"},
                outputs={str(path_output): "-c copy -framerate {}".format(fps)}
            )
            print("Running conversion command... [{}]".format(ff.cmd))
            ff.run()
        except ffmpy3.FFExecutableNotFoundError:
            print("FFMPEG executable not found!")
        except ffmpy3.FFRuntimeError:
            print("FFMPEG runtime error!")
        print("Video conversion complete!")
    except ImportError:
        print("Module ffmpy3 not fouund!")
    except:
        print("Unknown error in convert_to_mp4!")

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

_save_choices = ("color", "mono", "depth") # TODO: IMU/ToF

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings", type=str, help="Path where to store the captured data")
parser.add_argument('-s', '--save', default=["color", "mono"], nargs="+", choices=_save_choices,
                    help="Choose which streams to save. Default: %(default)s")
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
# TODO: make camera resolutions configrable

args = parser.parse_args()

def create_folder(path, mxid):
    i = 0
    while True:
        i += 1
        recordings_path = Path(path) / f"{i}-{str(mxid)}"
        if not recordings_path.is_dir():
            recordings_path.mkdir(parents=True, exist_ok=False)
            return recordings_path

def create_pipeline(save, fps):
    pipeline = dai.Pipeline()

    if "color" in save:
        rgb = pipeline.createColorCamera()
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setFps(fps)

        rgb_encoder = pipeline.createVideoEncoder()
        rgb_encoder.setDefaultProfilePreset(rgb.getVideoSize(), rgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        # rgb_encoder.setLossless(True)
        rgb.video.link(rgb_encoder.input)

        # Create output for the rgb
        rgbOut = pipeline.createXLinkOut()
        rgbOut.setStreamName("color")
        rgb_encoder.bitstream.link(rgbOut.input)

    if "mono" or "depth" in save:
        # Create mono cameras
        left = pipeline.createMonoCamera()
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setFps(fps)

        right = pipeline.createMonoCamera()
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setFps(fps)

        stereo = pipeline.createStereoDepth()
        stereo.initialConfig.setConfidenceThreshold(240)
        stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(False)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        # stereo.setRectification(False)
        # stereo.setRectifyMirrorFrame(False)

        left.out.link(stereo.left)
        right.out.link(stereo.right)

        if "depth" in save:
            depthOut = pipeline.createXLinkOut()
            depthOut.setStreamName("depth")
            stereo.depth.link(depthOut.input)

        # Create output
        if "mono" in save:
            left_encoder = pipeline.createVideoEncoder()
            left_encoder.setDefaultProfilePreset(left.getResolutionSize(), left.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            # left_encoder.setLossless(True)
            stereo.rectifiedLeft.link(left_encoder.input)
            # Create XLink output for left MJPEG stream
            leftOut = pipeline.createXLinkOut()
            leftOut.setStreamName("left")
            left_encoder.bitstream.link(leftOut.input)

            right_encoder = pipeline.createVideoEncoder()
            right_encoder.setDefaultProfilePreset(right.getResolutionSize(), right.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            # right_encoder.setLossless(True)
            stereo.rectifiedRight.link(right_encoder.input)
            # Create XLink output for right MJPEG stream
            rightOut = pipeline.createXLinkOut()
            rightOut.setStreamName("right")
            right_encoder.bitstream.link(rightOut.input)

    return pipeline

def store_frames(in_q, save, path):
    files = {}
    for stream_name in save:
        files[stream_name] = open(str(path / f"{stream_name}.mjpeg"), 'wb')

    while True:
        try:
            frames = in_q.get()
            if frames is None:
                break
            for name in frames:
                frames[name].tofile(files[name])
        except KeyboardInterrupt:
            break
    # Close all files
    for name in files:
        files[name].close()

    if True:
        print("Converting .mjpeg to .mp4")
        for stream_name in save:
            convert_to_mp4((path / f"{stream_name}.mjpeg"), args.fps)
    print('Exiting store frame process')

# Pipeline defined, now the device is connected to
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")

    devices = []

    for device_info in device_infos:
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = True
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))
        mxid = device.getMxId()
        stereo = 1 < len(device.getConnectedCameras())


        save = list(args.save)
        if not stereo: # If device doesn't have stereo camera pair
            if "mono" in save:
                save.remove("mono")
            if "depth" in save:
                save.remove("depth")
        device.startPipeline(create_pipeline(save, args.fps))

        streams = []
        if "color" in save: streams.append("color")
        if "depth" in save: streams.append("depth")
        if "mono" in save:
            streams.append("left")
            streams.append("right")

        frame_q = Queue(20)
        store_p = Process(target=store_frames, args=(frame_q, streams, create_folder(args.path, mxid)))
        store_p.start()

        queues = []
        for stream in streams:
            queues.append({
                'q': device.getOutputQueue(name=stream, maxSize=10, blocking=False),
                'msgs': [],
                'name': stream
            })

        devices.append({
            'queue': queues,
            'mx': mxid,
            'frame_q': frame_q, # Python Queue object
            'process': store_p
        })


    def check_sync(queues, timestamp):
        matching_frames = []
        for q in queues:
            for i, msg in enumerate(q['msgs']):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # So below 17ms @ 30 FPS => frames are in sync
                if time_diff <= timedelta(milliseconds=math.ceil(500 / args.fps)):
                    matching_frames.append(i)
                    break

        if len(matching_frames) == len(queues):
            # We have all frames synced. Remove the excess ones
            for i, q in enumerate(queues):
                q['msgs'] = q['msgs'][matching_frames[i]:]
            return True
        else:
            return False

    fps = FPSHandler()
    queues = [q for device in devices for q in device['queue']]
    while True:
        for q in queues:
            new_msg = q['q'].tryGet()
            if new_msg is not None:
                q['msgs'].append(new_msg)
                if check_sync(queues, new_msg.getTimestamp()):
                    fps.next_iter()
                    print("FPS:", fps.fps())
                    # print('frames synced')
                    for device in devices:
                        frames = {}
                        for stream in device['queue']:
                            frames[stream['name']] = stream['msgs'].pop(0).getData()
                            # cv2.imshow(f"{stream['name']} - {device['mx']}", cv2.imdecode(frames[stream['name']], cv2.IMREAD_UNCHANGED))
                        # print('For mx', device['mx'], 'frames')
                        # print('frames', frames)
                        device['frame_q'].put(frames)
        if cv2.waitKey(1) == ord('q'):
            break

    for device in devices:
        device['frame_q'].put(None)
        time.sleep(0.01) # Wait 10ms for process to close all files
        device['process'].join() # Terminate the process

