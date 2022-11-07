import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera
from pose import get_keypoints, get_personwise_keypoints, get_valid_pairs

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

NN_HEIGHT, NN_WIDTH = 256, 456


def decode(nn_data):
    heatmaps = np.array(nn_data.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
    pafs = np.array(nn_data.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
    heatmaps = heatmaps.astype('float32')
    pafs = pafs.astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    for row in range(18):
        prob_map = outputs[0, row, :, :]
        prob_map = cv2.resize(prob_map, (NN_WIDTH, NN_HEIGHT))  # (456, 256)
        keypoints = get_keypoints(prob_map, 0.3)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = get_valid_pairs(outputs, w=NN_WIDTH, h=NN_HEIGHT, detected_keypoints=new_keypoints)
    new_personwise_keypoints = get_personwise_keypoints(valid_pairs, invalid_pairs, new_keypoints_list)

    return new_keypoints, new_keypoints_list, new_personwise_keypoints


def callback(packet):
    nn_data = packet.img_detections
    frame = packet.frame

    keypoints, keypoints_list, personwise_keypoints = decode(nn_data)

    scale_factor = frame.shape[0] / NN_HEIGHT
    offset_w = int(frame.shape[1] - NN_WIDTH * scale_factor) // 2

    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

    for i in range(18):
        for j in range(len(keypoints[i])):
            cv2.circle(frame, scale(keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)

    for i in range(17):
        for n in range(len(personwise_keypoints)):
            index = personwise_keypoints[n][np.array(POSE_PAIRS[i])]

            if -1 in index:
                continue

            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)

    cv2.imshow('Human pose estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    blob_path = blobconverter.from_zoo(name='human-pose-estimation-0001', shaves=6)

    human_pose_nn = oak.create_nn(blob_path, color)

    oak.callback(human_pose_nn, callback)
    oak.start(blocking=True)
