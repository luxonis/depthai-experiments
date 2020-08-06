import ctypes
import json
import time
from pathlib import Path

import cv2
from depthai_utils import DepthAI
from modules import PersonTrackerDebug, PersonTracker
from multiprocessing import Process, Manager

debug = True

d = DepthAI(threshold=0.4)
pt = PersonTrackerDebug() if debug else PersonTracker()
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


for frame, results in d.run():
    total = pt.parse(frame, results)
    shared_results.value = json.dumps(pt.get_directions())

    if debug:
        for left, top, right, bottom in results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        print(pt.get_directions())
        cv2.imshow('previewout', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            pt.__init__()


del d
