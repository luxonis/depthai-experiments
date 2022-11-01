#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
parser.add_argument('-d', '--draw', default=False, action='store_true',
                    help='Draw on frames the sequence number and timestamp')
parser.add_argument('-v', '--verbose', default=0, action='count',
                    help='Verbose, -vv for more verbosity')
parser.add_argument('-t', '--dev_timestamp', default=False, action='store_true',
                    help='Get device timestamps, not synced to host. For debug')
parser.add_argument('-hw', '--hardware_sync', const='ext', choices={'ext', 'cam', 'gpio'}, nargs="?",
                    help='Enable hardware sync (instead of software sync) for capture.\n'
                    'Optional parameter to configure sync type:\n'
                    '  ext  - external, signal rate should be less than configured FPS\n'
                    '         [default option if unspecified]\n'
                    '         Note: signal should be provided inverted on M8 connector,\n'
                    '               ideally with a short active time (under 1ms)\n'
                    '  cam  - sync signal outputted by Left camera, internally on device\n'
                    '         (no output on PoE M8 connector)\n'
                    '  gpio - generated internally on device by a SoC GPIO at about 5Hz rate')
args = parser.parse_args()

cam_list = ['left', 'rgb', 'right']
cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.RGB,
    'left' : dai.CameraBoardSocket.LEFT,
    'right': dai.CameraBoardSocket.RIGHT,
}
cam_instance = {
    'rgb'  : 0,
    'left' : 1,
    'right': 2,
}

# Start defining a pipeline
pipeline = dai.Pipeline()

cam = {}
xout = {}
for c in cam_list:
    xout[c] = pipeline.create(dai.node.XLinkOut)
    xout[c].setStreamName(c)
    if c == 'rgb':
        cam[c] = pipeline.create(dai.node.ColorCamera)
        cam[c].setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam[c].setIspScale(720, 1080)  # 1920x1080 -> 1280x720
        cam[c].isp.link(xout[c].input)
    else:
        cam[c] = pipeline.create(dai.node.MonoCamera)
        cam[c].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam[c].out.link(xout[c].input)
    cam[c].setBoardSocket(cam_socket_opts[c])
    cam[c].setFps(args.fps)

# Note1: IMX378 (RGB camera) doesn't support `setExternalTrigger`
# Note2: `.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)` for OV9282
#         (left, right cams) has very tight timing requirements
if args.hardware_sync in ['ext', 'gpio']:
    cam['rgb'  ].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    if 0:
        cam['left' ].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
        cam['right'].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    else:
        cam['left' ].initialControl.setExternalTrigger(1, 0) # (2, 1)
        cam['right'].initialControl.setExternalTrigger(1, 0)
        print("WARNING: Configured FPS ignored for L/R...")
        cam['left' ].setFps(60)
        cam['right'].setFps(60)
    if args.hardware_sync == 'gpio':
        scriptTrigger = pipeline.create(dai.node.Script)
        scriptTrigger.setProcessor(dai.ProcessorType.LEON_CSS)
        scriptTrigger.setScript("""
            import GPIO, time
            gpioNum = 41  # FSIN pins connected to this SoC GPIO
            activeLevel = GPIO.HIGH
            timeActive   = 0.001
            timeInactive = 0.199

            GPIO.setup(gpioNum, GPIO.OUT)
            if 0: 
                # For NG9097 R1M1E1 only (just few samples, no mass production),
                # no optocoupler and where bi-dir is possible
                fsyncDirGpio = 39
                GPIO.setup(fsyncDirGpio, GPIO.OUT)
                GPIO.write(fsyncDirGpio, GPIO.HIGH) # FSYNC output on M8 connector
            while True:
                time.sleep(timeInactive)

                GPIO.write(gpioNum, activeLevel)
                time.sleep(timeActive)

                GPIO.write(gpioNum, not activeLevel)
        """)
if args.hardware_sync == 'cam':
    cam['rgb'  ].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    cam['left' ].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.OUTPUT)
    cam['right'].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    cam['rgb'  ].setFps(args.fps + 0.1) # Note: slightly larger than for L/R
    cam['left' ].setFps(args.fps)
    cam['right'].setFps(args.fps)

def get_seq(packet):
    return packet.getSequenceNum()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    allowed_instances = [0, 1, 2]  # Center (0) & Left (1) & Right (2)

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self.allowed_instances:
            seq_key = get_seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    tstart = time.monotonic()  # applied as an offset

    saveidx = 0

    q = {}
    for c in cam_list:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
    ps = PairingSystem()

    window = ' + '.join(cam_list)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, (3*1280//2, 720//2))

    if args.verbose > 0:
        print("   Seq  Left_tstamp  RGB-Left  Right-Left  Dropped")
        print("   num    [seconds]  diff[ms]    diff[ms]  delta")

    seq_prev = -1
    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        for c in cam_list:
            ps.add_packet(q[c].tryGet())

        for synced in ps.get_pairs():
            frame, seqnum, tstamp = {}, {}, {}
            for c in cam_list:
                pkt = synced[cam_instance[c]]
                frame[c] = pkt.getCvFrame()
                if c != 'rgb': 
                    frame[c] = cv2.cvtColor(frame[c], cv2.COLOR_GRAY2BGR)
                seqnum[c] = pkt.getSequenceNum()
                if args.dev_timestamp:
                    tstamp[c] = pkt.getTimestampDevice().total_seconds()
                else:
                    tstamp[c] = pkt.getTimestamp().total_seconds() - tstart
                if args.draw:
                    text = '{:1d}'.format(seqnum[c]) + '  {:.6f}'.format(tstamp[c])
                    cv2.putText(frame[c], text, (8,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 8, cv2.LINE_AA)
                    cv2.putText(frame[c], text, (8,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            if not(seqnum['rgb'] == seqnum['left'] == seqnum['right']):
                print('ERROR: out of sync!!!')
            if args.verbose > 0:
                seq = seqnum['left']
                seq_diff = seq - seq_prev
                seq_prev = seq
                rgb_left_diff   = (tstamp['rgb']   - tstamp['left']) * 1000
                right_left_diff = (tstamp['right'] - tstamp['left']) * 1000
                if (args.verbose > 1 or seq_diff != 1
                                     or abs(rgb_left_diff)   > 0.15
                                     or abs(right_left_diff) > 0.05):
                    print('{:6d} {:12.6f} {:9.3f}   {:9.3f}'.format(
                          seq, tstamp['left'], rgb_left_diff, right_left_diff), end='')
                    if seq_diff != 1: print('   ', seq_diff - 1, end='')
                    print()
            frame_final = np.hstack(([frame[c] for c in cam_list]))
            cv2.imshow(window, frame_final)

            key = cv2.waitKey(1)
            if key == ord('c'):
                filename = f'capture_{saveidx:04}.png'
                cv2.imwrite(filename, frame_final)
                print('Saved to:', filename)
                saveidx += 1
            elif key == ord('q'):
                quit()
