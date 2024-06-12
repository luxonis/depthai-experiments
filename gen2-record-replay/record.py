#!/usr/bin/env python3
import depthai as dai
from pathlib import Path
import signal
import argparse
from depthai_sdk.components.parser import parse_camera_socket
from depthai_sdk import OakCamera, ArgsParser, RecordType
import threading

_quality_choices = ['BEST', 'HIGH', 'MEDIUM', 'LOW']

def checkQuality(value: str):
    if value.upper() in _quality_choices:
        return value
    elif value.isdigit():
        num = int(value)
        if 0 <= num <= 100:
            return num
    raise argparse.ArgumentTypeError(f"{value} is not a valid quality. Either {'/'.join(_quality_choices)}, or a number 0-100.")

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings", type=str, help="Path where to store the captured data")
parser.add_argument('-save', '--save', default=["color", "left", "right"], nargs="+", help="Choose which streams to save. Default: %(default)s")
# parser.add_argument('-fc', '--frame_cnt', type=int, default=-1,
#                     help='Number of frames to record. Record until stopped by default.')
parser.add_argument('-q', '--quality', default="HIGH", type=checkQuality,
                    help='Selects the quality of the recording. Default: %(default)s')
parser.add_argument('-type', '--type', default="VIDEO", help="Recording type. Default: %(default)s", choices=['VIDEO', 'ROSBAG', 'DB3'])
parser.add_argument('--disable_preview', action='store_true', help="Disable preview output to reduce resource usage. By default, all streams are displayed.")

args = ArgsParser.parseArgs(parser)

sockets = []
for i, stream in enumerate(args['save']):
    stream: str = stream.lower()
    args['save'][i] = stream
    if stream in ['disparity', 'depth']:
        # All good
        continue
    sockets.append(parse_camera_socket(stream))

if args['rgbFps'] != args['monoFps']:
    raise ValueError('RGB and MONO FPS must be the same when recording for now!')

def create_cam(socket: dai.CameraBoardSocket):
    if args['quality'] == 'LOW':
        cam = oak.create_camera(socket, encode=dai.VideoEncoderProperties.Profile.H265_MAIN)
        cam.config_encoder_h26x(bitrate_kbps=10000)
        return cam

    cam = oak.create_camera(socket, encode=dai.VideoEncoderProperties.Profile.MJPEG)

    if args['quality'].isdigit():
        cam.config_encoder_mjpeg(quality=int(args['quality']))
    elif args['quality'] == 'BEST':
        cam.config_encoder_mjpeg(lossless=True)
    elif args['quality'] == 'HIGH':
        cam.config_encoder_mjpeg(quality=97)
    elif args['quality'] == 'MEDIUM':
        cam.config_encoder_mjpeg(quality=93)
    return cam

save_path = Path(__file__).parent / args['path']

print('save path', save_path)

with OakCamera(args=args) as oak:
    calib = oak.device.readCalibrationOrDefault()

    recording_list = []

    if 'disparity' in args['save'] or 'depth' in args['save']:
        left_socket = calib.getStereoLeftCameraId()
        right_socket = calib.getStereoRightCameraId()

        left = create_cam(left_socket)
        right = create_cam(right_socket)

        if left_socket in sockets:
            sockets.remove(left_socket)
            recording_list.append(left)
        if right_socket in sockets:
            sockets.remove(right_socket)
            recording_list.append(right)

        stereo = oak.create_stereo(left=left, right=right)

        if 'disparity' in args['save']:
            recording_list.append(stereo.out.disparity)
        if 'depth' in args['save']:
            recording_list.append(stereo.out.depth)

    for socket in sockets:
        cam = create_cam(socket)
        recording_list.append(cam)
        if not args['disable_preview']:
            oak.visualize(cam, scale=2/3, fps=True)

    oak.record(recording_list, path=save_path, record_type=getattr(RecordType, args['type']))

    oak.start(blocking=False)

    quitEvent = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: quitEvent.set())
    print("\nRecording started. Press 'Ctrl+C' to stop.")

    while oak.running() and not quitEvent.is_set():
        oak.poll()
