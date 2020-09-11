import math
from pathlib import Path

import numpy as np

from visualizer import initialize_OpenGL, get_vector_direction, get_vector_intersection, start_OpenGL
import cv2
import depthai

device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": str(
            Path('./models/face-detection-retail-0004/face-detection-retail-0004.blob').resolve().absolute()),
        "blob_file_config": str(
            Path('./models/face-detection-retail-0004/face-detection-retail-0004.json').resolve().absolute()),
        'blob_file2': str(Path(
            './models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob').resolve().absolute()),
        'blob_file_config2': str(Path(
            './models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.json').resolve().absolute()),
        'camera_input': "left_right",
        'NN_engines': 2,
        'shaves': 14,
        'cmx_slices': 14,
    }
})

if p is None:
    raise RuntimeError("Error initializing pipelne")


def get_landmark_3d(landmark):
    focal_length = 842
    landmark_norm = 0.5 - np.array(landmark)

    # image size
    landmark_image_coord = landmark_norm * 640

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D


initialize_OpenGL()

entries_prev = {}
left_landmarks = []
right_landmarks = []

left_camera_position = (0.107, -0.038, 0.008)
right_camera_position = (0.109, 0.039, 0.008)
cameras = ((0.107, -0.038, 0.008), (0.109, 0.039, 0.008))

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
            landmarks_3d = list(map(get_landmark_3d, landmarks_pairs))
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
                    "landmarks_3d": landmarks_3d,
                })

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            cam = packet.getMetadata().getCameraName()
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            landmarks_frame = np.zeros((img_h, img_w, 3), np.uint8)

            for e in entries_prev.get(cam, []):
                left = int(e['left'] * img_w)
                top = int(e['top'] * img_h)
                right = int(e['right'] * img_w)
                bottom = int(e['bottom'] * img_h)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(landmarks_frame, (left, top), (right, bottom), (0, 0, 255), 2)

                face_width = int(right - left)
                face_height = int(bottom - top)
                for land_x, land_y in e['landmarks']:
                    x = left + int(land_x * face_width)
                    y = top + int(land_y * face_height)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0))
                    cv2.circle(landmarks_frame, (x, y), 4, (255, 0, 0))

                if cam == "left":
                    left_landmarks = e['landmarks_3d']
                else:
                    right_landmarks = e['landmarks_3d']

            cv2.imshow(f'previewout-{packet.getMetadata().getCameraName()}', frame)
            cv2.imshow(packet.getMetadata().getCameraName(), landmarks_frame)

            if len(left_landmarks) > 0 and len(right_landmarks) > 0:
                mid_intersects = []

                for i in range(len(left_landmarks)):
                    left_vector = get_vector_direction(left_camera_position, left_landmarks[i])
                    right_vector = get_vector_direction(right_camera_position, right_landmarks[i])
                    intersection_landmark = get_vector_intersection(left_vector, left_camera_position, right_vector,
                                                                    right_camera_position)
                    mid_intersects.append(intersection_landmark)

                start_OpenGL(mid_intersects, cameras, left_landmarks, right_landmarks)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device
