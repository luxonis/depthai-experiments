import depthai as dai
import cv2
import numpy as np

from pose_utils import getKeypoints, getValidPairs, getPersonwiseKeypoints

COLORS = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

class HumanPose(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "HumanPose":
        self.link_args(preview, nn)
        return self

    # Preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert(isinstance(preview, dai.ImgFrame))
        frame = preview.getCvFrame()

        pafs = nn.getTensor("Mconv7_stage2_L1").flatten().reshape((1, 38, 32, 57)).astype(np.float32)
        heatmaps = nn.getTensor("Mconv7_stage2_L2").flatten().reshape((1, 19, 32, 57)).astype(np.float32)
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            prob_map = outputs[0, row, :, :]
            prob_map = cv2.resize(prob_map, (456, 256))
            new_keypoints = getKeypoints(prob_map, 0.3)
            keypoints_list = np.vstack([keypoints_list, *new_keypoints])
            keypoints_with_id = []

            for i in range(len(new_keypoints)):
                keypoints_with_id.append(new_keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w=456, h=256, detected_keypoints=keypoints)
        personwise_keypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

        if keypoints_list is not None and keypoints is not None and personwise_keypoints is not None:
            scale_factor = frame.shape[0] / 256
            offset_w = int(frame.shape[1] - 456 * scale_factor) // 2

            def scale(point):
                return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

            for i in range(18):
                for j in range(len(keypoints[i])):
                    cv2.circle(frame, scale(keypoints[i][j][0:2]), 5, COLORS[i], -1, cv2.LINE_AA)
            for i in range(17):
                for n in range(len(personwise_keypoints)):
                    index = personwise_keypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), COLORS[i], 3, cv2.LINE_AA)

        preview.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
        self.output.send(preview)
