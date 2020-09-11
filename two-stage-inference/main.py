from pathlib import Path

import cv2
import depthai

device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": str(Path('./models/face-detection-retail-0004/face-detection-retail-0004.blob').resolve().absolute()),
        "blob_file_config": str(Path('./models/face-detection-retail-0004/face-detection-retail-0004.json').resolve().absolute()),
        'blob_file2': str(Path('./models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob').resolve().absolute()),
        'blob_file_config2': str(Path('./models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.json').resolve().absolute()),
        'camera_input': "left_right",
        'NN_engines': 2,
        'shaves': 14,
        'cmx_slices': 14,
    }
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

entries_prev = {}

while True:
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets(True)

    for nnet_packet in nnet_packets:
        cam = nnet_packet.getMetadata().getCameraName()
        entries_prev[cam] = []
        for e in nnet_packet.entries():
            if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
                break

            landmarks_raw = [e[1][i] for i in range(len(e[1]))]
            landmarks_pairs = list(zip(*[iter(landmarks_raw)] * 2))
            if e[0]['confidence'] > 0.5:
                entries_prev[cam].append({
                    "id": e[0]["id"],
                    "label": e[0]["label"],
                    "confidence": e[0]["confidence"],
                    "left": e[0]["left"],
                    "top": e[0]["top"],
                    "right": e[0]["right"],
                    "bottom": e[0]["bottom"],
                    "landmarks": landmarks_pairs,
                })

    for packet in data_packets:
        cam = packet.getMetadata().getCameraName()
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for e in entries_prev.get(cam, []):
                left = int(e['left'] * img_w)
                top = int(e['top'] * img_h)
                right = int(e['right'] * img_w)
                bottom = int(e['bottom'] * img_h)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                face_width = int(right - left)
                face_height = int(bottom - top)
                for land_x, land_y in e['landmarks']:
                    x = left + int(land_x * face_width)
                    y = top + int(land_y * face_height)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0))

            cv2.imshow(f'previewout-{cam}', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device