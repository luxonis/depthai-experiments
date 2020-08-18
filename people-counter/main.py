import argparse
import ctypes
import json
import time
import os
from pathlib import Path

import cv2
from modules import PeopleCounter
from depthai_utils import DepthAI
from multiprocessing import Process, Manager

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='person_detection_retail_0013', choices=list(os.listdir('./models')))
args = parser.parse_args()

debug = True
shared_results = Manager().Value(ctypes.c_wchar_p, '{}')

#  https://stackoverflow.com/a/44599922/5494277
def append_to_json(_dict, path):
    with open(path, 'ab+') as f:
        f.seek(0, 2)  # Go to the end of file
        if f.tell() == 0:  # Check if file is empty
            f.write(json.dumps([_dict]).encode())  # If empty, write an array
        else:
            f.seek(-1, 2)
            f.truncate()  # Remove the last character, open the array
            f.write(' , '.encode())  # Write the separator
            f.write(json.dumps(_dict).encode())  # Dump the dictionary
            f.write(']'.encode())


def store(payload):
    storage_path = Path('results.json')
    while True:
        try:
            loaded = json.loads(payload.value)
            if len(loaded) == 0:
                continue
            loaded['timestamp'] = int(time.time())
            append_to_json(loaded, storage_path)
            time.sleep(1)
        except:
            pass

p = Process(target=store, args=(shared_results, ))
p.daemon = True
p.start()

d = DepthAI(model_name=args.model, threshold=0.5)
pc = PeopleCounter()

for frame, results in d.run():
    total = pc.parse(results)
    shared_results.value = json.dumps({"count": total})

    if debug:
        print(f"Detected: {total}")
        cv2.putText(frame, f"Detected: {total}", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        for left, top, right, bottom in results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow('previewout', frame)

        if cv2.waitKey(1) == ord('q'):
            break

del d
