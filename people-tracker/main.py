from pathlib import Path

import consts.resource_paths
import cv2
import depthai

debug = True

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")

p = depthai.create_pipeline(config={
    "streams": ["previewout", "object_tracker"],
    "ai": {
        "blob_file": str(Path('model/model.blob').absolute()),
        "blob_file_config": str(Path('model/config.json').absolute()),
    },
    'ot': {
        'max_tracklets': 20,
        'confidence_threshold': 0.3,
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

entries_prev = []
tracklets = None
positions = {}

while True:
    for packet in p.get_available_data_packets():
        if packet.stream_name == 'object_tracker':
            tracklets = packet.getObjectTracker()
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            traklets_nr = tracklets.getNrTracklets() if tracklets is not None else 0

            for i in range(traklets_nr):
                tracklet = tracklets.getTracklet(i)
                left = tracklet.getLeftCoord()
                top = tracklet.getTopCoord()
                right = tracklet.getRightCoord()
                bottom = tracklet.getBottomCoord()

                middle_pt = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))

                if tracklet.getId() not in positions:
                    positions[tracklet.getId()] = []

                positions[tracklet.getId()].append(middle_pt[1])

                if debug:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))
                    cv2.circle(frame, middle_pt, 0, (255, 0, 0), -1)
                    cv2.putText(frame, f"ID {tracklet.getId()}", middle_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, tracklet.getStatus(), (left, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            up, down = (0, 0)
            for item_positions in positions.values():
                if len(item_positions) < 3:
                    continue
                y_max_min_diff = item_positions.index(max(item_positions)) - item_positions.index(min(item_positions))
                if y_max_min_diff > 0:
                    down += 1
                else:
                    up += 1

            print(f"Up: {up}, Down: {down}")
            if debug:
                cv2.putText(frame, f"Up: {up}", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Down: {down}", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
depthai.deinit_device()