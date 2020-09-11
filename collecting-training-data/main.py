import argparse
from pathlib import Path
from multiprocessing import Process
from uuid import uuid4

import cv2
import depthai

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.03, type=float, help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
args = parser.parse_args()

device = depthai.Device('', False)

dest = Path(args.path).resolve().absolute()
if dest.exists() and len(list(dest.glob('*'))) != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {len(list(dest.glob('*')))} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

p = device.create_pipeline(config={
    "streams": ["left", "right", "previewout", "disparity_color"],
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

latest_left = None
lr_pairs = {}
previewouts = {}
procs = []

# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold
def seq(packet):
    return packet.getMetadata().getSequenceNum()
def tst(packet):
    return packet.getMetadata().getTimestamp()
def store_frames(left, right, rgb, disparity):
    global procs
    frames_path = dest / Path(str(uuid4()))
    frames_path.mkdir(parents=False, exist_ok=False)
    new_procs = [
        Process(target=cv2.imwrite, args=(str(frames_path / Path("left.png")), left)),
        Process(target=cv2.imwrite, args=(str(frames_path / Path("right.png")), right)),
        Process(target=cv2.imwrite, args=(str(frames_path / Path("rgb.png")), rgb)),
        Process(target=cv2.imwrite, args=(str(frames_path / Path("disparity.png")), disparity)),
    ]
    for proc in new_procs:
        proc.start()
    procs += new_procs

while True:
    data_packets = p.get_available_data_packets()

    for packet in data_packets:
        print(packet.stream_name)
        print(packet.getMetadata().getTimestamp(), packet.getMetadata().getSequenceNum(), packet.stream_name)
        if packet.stream_name == "left":
            latest_left = packet
        elif packet.stream_name == "right" and latest_left is not None and seq(latest_left) == seq(packet):
            lr_pairs[seq(packet)] = (latest_left, packet)
        elif packet.stream_name == 'previewout':
            previewouts[step_norm(tst(packet))] = packet
        elif packet.stream_name == "disparity_color":
            if seq(packet) in lr_pairs and step_norm(tst(packet)) in previewouts:
                left, right = lr_pairs[seq(packet)]
                previewout = previewouts[step_norm(tst(left))]

                data = previewout.getData()
                data0 = data[0, :, :]
                data1 = data[1, :, :]
                data2 = data[2, :, :]
                rgb = cv2.merge([data0, data1, data2])

                store_frames(left.getData(), right.getData(), rgb, packet.getData())
                cv2.imshow('left', left.getData())
                cv2.imshow('right', right.getData())
                cv2.imshow('previewout', rgb)
                cv2.imshow('disparity_color', packet.getData())
            else:
                for key in list(lr_pairs.keys()):
                    if key < seq(packet):
                        del lr_pairs[key]
                for key in list(previewouts.keys()):
                    if key < tst(packet):
                        del previewouts[key]

    if cv2.waitKey(1) == ord('q'):
        break

for proc in procs:
    proc.join()
del p
del device