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
        'camera_input': "left_right",
        'NN_engines': 2,
        'shaves': 14,
        'cmx_slices': 14,
    }
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

detections = {}

while True:
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets(True)

    for nnet_packet in nnet_packets:
        cam = nnet_packet.getMetadata().getCameraName()
        detections[cam] = {
            'face': nnet_packet.getDetectedObjects(),
            'land': list(zip(*[iter(nnet_packet.get_tensor(1).reshape((10, )))] * 2))
        }

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

            for detection in detections.get(cam, {}).get('face', []):
                left, top = int(detection.x_min * img_w), int(detection.y_min * img_h)
                right, bottom = int(detection.x_max * img_w), int(detection.y_max * img_h)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                face_width = int(right - left)
                face_height = int(bottom - top)

                for land_x, land_y in detections.get(cam, {}).get('land', []):
                    x = left + int(land_x * face_width)
                    y = top + int(land_y * face_height)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0))

            cv2.imshow(f'previewout-{cam}', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device