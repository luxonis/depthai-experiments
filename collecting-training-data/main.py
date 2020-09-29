import argparse
from pathlib import Path
from multiprocessing import Process
from time import time
from uuid import uuid4

import cv2
import depthai

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.03, type=float,
                    help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
parser.add_argument('-nd', '--no-debug', dest="prod", action='store_true', default=False, help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")
parser.add_argument('-af', '--autofocus', type=str, default=None, help="Set AutoFocus mode of the RGB camera", choices=list(filter(lambda name: name.startswith("AF_"), vars(depthai.AutofocusMode))))
args = parser.parse_args()

device = depthai.Device('', False)
device.request_af_mode(getattr(depthai.AutofocusMode, args.autofocus, depthai.AutofocusMode.AF_MODE_AUTO))

dest = Path(args.path).resolve().absolute()
if dest.exists() and len(list(dest.glob('*'))) != 0 and not args.dirty:
    raise ValueError(
        f"Path {dest} contains {len(list(dest.glob('*')))} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

p = device.create_pipeline(config={
    "streams": ["left", "right", "color", "disparity_color"],
    "ai": {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
        "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute())
    },
    'camera': {
        'mono': {
            'resolution_h': 720, 'fps': 30
        },
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

procs = []


# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold


def seq(packet):
    return packet.getMetadata().getSequenceNum()


def tst(packet):
    return packet.getMetadata().getTimestamp()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


def extract_color_stream(item):
    data = item.getData()
    meta = item.getMetadata()
    w = int(meta.getFrameWidth())
    h = int(meta.getFrameHeight())
    yuv420p = data.reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


extract_frame = {
    "left": lambda item: item.getData(),
    "right": lambda item: item.getData(),
    "disparity_color": lambda item: item.getData(),
    "color": extract_color_stream,
}


def store_frames(frames_dict):
    global procs
    frames_path = dest / Path(str(uuid4()))
    frames_path.mkdir(parents=False, exist_ok=False)
    new_procs = [
        Process(
            target=cv2.imwrite,
            args=(
                str(frames_path / Path(f"{stream_name}.png")),
                extract_frame[stream_name](frames_dict[stream_name])
            )
        )
        for stream_name in frames_dict
    ]
    for proc in new_procs:
        proc.start()
    procs += new_procs


class PairingSystem:
    seq_streams = ["left", "right", "disparity_color"]
    ts_streams = ["color"]
    seq_ts_mapping_stream = "left"  # which stream's packet will be responsible for both ts and seq matching

    def __init__(self):
        self.ts_packets = {}
        self.seq_packets = {}
        self.last_paired_ts = None
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet.stream_name in self.seq_streams:
            seq_key = seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.stream_name: packet
            }
        elif packet.stream_name in self.ts_streams:
            ts_key = step_norm(tst(packet))
            self.ts_packets[ts_key] = {
                **self.ts_packets.get(ts_key, {}),
                packet.stream_name: packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.seq_streams):
                ts_key = step_norm(tst(self.seq_packets[key][self.seq_ts_mapping_stream]))
                if ts_key in self.ts_packets and has_keys(self.ts_packets[ts_key], self.ts_streams):
                    results.append({
                        **self.seq_packets[key],
                        **self.ts_packets[ts_key]
                    })
                    self.last_paired_seq = key
                    self.last_paired_ts = ts_key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
        for key in list(self.ts_packets.keys()):
            if key <= self.last_paired_ts:
                del self.ts_packets[key]


ps = PairingSystem()
start_ts = time()

while True:
    data_packets = p.get_available_data_packets()

    for packet in data_packets:
        if not args.prod:
            print(packet.stream_name)
            print(packet.getMetadata().getTimestamp(), packet.getMetadata().getSequenceNum())
        ps.add_packet(packet)
        pairs = ps.get_pairs()
        for pair in pairs:
            if not args.prod:
                for stream_name in pair:
                    frame = extract_frame[stream_name](pair[stream_name])
                    cv2.imshow(stream_name, frame)
            store_frames(pair)

    if not args.prod and cv2.waitKey(1) == ord('q'):
        break

    if time() - start_ts > args.time:
        break

for proc in procs:
    proc.join()
del p
del device
