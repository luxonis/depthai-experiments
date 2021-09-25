import argparse
from os import pipe
from pathlib import Path
import cv2
import numpy as np
import depthai as dai
from tools import *
from imutils.video import FPS
import queue
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("-nd", "--no-debug", action="store_true", help="Prevent debug output")
parser.add_argument(
    "-cam", "--camera", action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)"
)
parser.add_argument(
    "-vid", "--video", type=str, help="Path to video file to be used for inference (conflicts with -cam)"
)
args = parser.parse_args()

debug = True
camera = True

# TODO: Uncomment
# if args.camera and args.video:
#     raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
# elif args.camera is False and args.video is None:
#     raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_tensor_result(packet):
    return {name: np.array(packet.getLayerFp16(name)) for name in [tensor.name for tensor in packet.getRaw().tensors]}


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def coordinate(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result


def draw_bbox(debug_frame, bbox, color):
    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        # cam.setPreviewSize(1080, 1080)
        cam.setPreviewSize(300, 300)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setBlobPath(
        str(Path("models/face-detection-retail-0004_openvino_2020_1_4shave.blob").resolve().absolute())
    )
    # Link ImageManip -> Face detection NN node
    face_det_manip.out.link(face_det_nn.input)

    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("face_nn")
    face_det_nn.out.link(face_det_xout.input)

    # Script node will take the output from the face detection NN as input
    image_manip_script = pipeline.create(dai.node.Script)
    image_manip_script.inputs["face_in"].setBlocking(False)
    image_manip_script.inputs["face_in"].setQueueSize(4)
    face_det_nn.out.link(image_manip_script.inputs["face_in"])

    image_manip_script.setScript(
        """
    # def limit_roi(nn_data):
    #     if nn_data[3] < 0: nn_data[3] = 0
    #     if nn_data[4] < 0: nn_data[4] = 0
    #     if nn_data[5] > 0.999: nn_data[5] = 0.999
    #     if nn_data[6] > 0.999: nn_data[6] = 0.999

    while True:
        face_dets = node.io['face_in'].get().detections
        # node.warn(f"Faces detected: {len(face_dets)}")
        for det in face_dets:
            # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(160, 160)
            cfg.setKeepAspectRatio(False)
            node.io['to_manip'].send(cfg)
    """
    )

    landmark_manip = pipeline.create(dai.node.ImageManip)
    landmark_manip.initialConfig.setResize(160, 160)
    landmark_manip.initialConfig.setKeepAspectRatio(False)
    landmark_manip.setWaitForConfigInput(False)
    image_manip_script.outputs["to_manip"].link(landmark_manip.inputConfig)

    if camera:
        # Use 1080x1080 full image for both NNs
        cam.preview.link(face_det_manip.inputImage)
        cam.preview.link(landmark_manip.inputImage)

    land_manip_xout = pipeline.create(dai.node.XLinkOut)
    land_manip_xout.setStreamName("land_manip_out")
    landmark_manip.out.link(land_manip_xout.input)

    landmark_nn = pipeline.create(dai.node.NeuralNetwork)
    landmark_nn.setBlobPath(str(Path("models/face_landmark_160x160_openvino_2020_1_4shave.blob").resolve().absolute()))
    landmark_manip.out.link(landmark_nn.input)

    lankmark_nn_xout = pipeline.create(dai.node.XLinkOut)
    lankmark_nn_xout.setStreamName("land_nn")
    landmark_nn.out.link(lankmark_nn_xout.input)

    return pipeline


print("Starting pipeline...")
with dai.Device(create_pipeline()) as device:
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)
    if camera:
        cam_out = device.getOutputQueue("cam_out", 4, False)

    face_q = device.getOutputQueue("face_nn", 4, False)
    landmark_q = device.getOutputQueue("land_nn", 4, False)

    # Has to be here, otherwise it crashes...
    landmark_manip_q = device.getOutputQueue("land_manip_out", 4, False)

    results = []
    bboxes = []

    fps = FPS()
    fps.start()

    # def should_run():
    # return cap.isOpened() if args.video else True

    def get_frame():
        return True, np.array(cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)

    try:
        while True:
            read_correctly, frame = get_frame()
            if not read_correctly or frame is None:
                break

            # land_manip_out = (
                # np.array(landmark_manip_q.get().getData()).reshape((3, 160, 160)).transpose(1, 2, 0).astype(np.uint8)
            # )
            # cv2.imshow("land_manip_out", land_manip_out)

            fps.update()
            debug_frame = frame.copy()

            det_in = face_q.tryGet()
            if det_in is not None:
                detections = det_in.detections
                for detection in detections:
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    draw_bbox(debug_frame, bbox, (255, 0, 0))

                    # If there is a face detected, there will also be a landmark
                    # inference result available soon, so we can wait for it
                    land68_data = landmark_q.get()
                    out = to_tensor_result(land68_data).get("StatefulPartitionedCall/strided_slice_2/Split.0")
                    face_frame = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    result = coordinate(face_frame, *out)

                    while not len(results) < len(detections) and len(results) > 0:
                        results.pop(0)
                    results.append({"bbox": bbox, "result": result, "ts": time.time()})

            # Display results for 0.2 seconds after the inference
            results = list(filter(lambda result: time.time() - result["ts"] < 0.2, results))

            if debug:
                if len(results) > 0:
                    try:
                        for curr_result in results:
                            bbox = curr_result["bbox"]
                            result = curr_result["result"]
                            face_bbox = bbox
                            face_bbox[1] -= 15
                            face_bbox[3] += 15
                            face_bbox[0] -= 15
                            face_bbox[2] += 15

                            draw_bbox(debug_frame, bbox, (0, 255, 0))
                            hand_points = []
                            # 17 Left eyebrow upper left corner/21 Left eyebrow right corner/22 Right eyebrow upper left corner/26 Right eyebrow upper right corner/36 Left eye upper left corner/39 Left eye upper right corner/42 Right eye upper left corner/
                            # 45 Upper right corner of the right eye/31 Upper left corner of the nose/35 Upper right corner of the nose/48 Upper left corner/54 Upper right corner of the mouth/57 Lower central corner of the mouth/8 Chin corner
                            # The coordinates are two points, so you have to multiply by 2.
                            hand_points.append((result[34] + face_bbox[0], result[35] + face_bbox[1]))
                            hand_points.append((result[42] + face_bbox[0], result[43] + face_bbox[1]))
                            hand_points.append((result[44] + face_bbox[0], result[45] + face_bbox[1]))
                            hand_points.append((result[52] + face_bbox[0], result[53] + face_bbox[1]))
                            hand_points.append((result[72] + face_bbox[0], result[73] + face_bbox[1]))
                            hand_points.append((result[78] + face_bbox[0], result[79] + face_bbox[1]))
                            hand_points.append((result[84] + face_bbox[0], result[85] + face_bbox[1]))
                            hand_points.append((result[90] + face_bbox[0], result[91] + face_bbox[1]))
                            hand_points.append((result[62] + face_bbox[0], result[63] + face_bbox[1]))
                            hand_points.append((result[70] + face_bbox[0], result[71] + face_bbox[1]))
                            hand_points.append((result[96] + face_bbox[0], result[97] + face_bbox[1]))
                            hand_points.append((result[108] + face_bbox[0], result[109] + face_bbox[1]))
                            hand_points.append((result[114] + face_bbox[0], result[115] + face_bbox[1]))
                            hand_points.append((result[16] + face_bbox[0], result[17] + face_bbox[1]))
                            for i in hand_points:
                                cv2.circle(debug_frame, (i[0], i[1]), 2, (255, 0, 0), thickness=1, lineType=8, shift=0)
                            reprojectdst, _, pitch, yaw, roll = get_head_pose(np.array(hand_points))

                            """
                            pitch > 0 Head down, < 0 look up
                            yaw > 0 Turn right < 0 Turn left
                            roll > 0 Tilt right, < 0 Tilt left
                            """
                            cv2.putText(
                                debug_frame,
                                "pitch:{:.2f}, yaw:{:.2f}, roll:{:.2f}".format(pitch, yaw, roll),
                                (face_bbox[0] - 30, face_bbox[1] - 30),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.45,
                                (255, 0, 0),
                            )

                            hand_attitude = np.array([abs(pitch), abs(yaw), abs(roll)])
                            max_index = np.argmax(hand_attitude)
                            if max_index == 0:
                                if pitch > 0:
                                    cv2.putText(
                                        debug_frame,
                                        "Head down",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                                else:
                                    cv2.putText(
                                        debug_frame,
                                        "look up",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                            elif max_index == 1:
                                if yaw > 0:
                                    cv2.putText(
                                        debug_frame,
                                        "Turn right",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                                else:
                                    cv2.putText(
                                        debug_frame,
                                        "Turn left",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                            elif max_index == 2:
                                if roll > 0:
                                    cv2.putText(
                                        debug_frame,
                                        "Tilt right",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                                else:
                                    cv2.putText(
                                        debug_frame,
                                        "Tilt left",
                                        (face_bbox[0], face_bbox[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (235, 10, 10),
                                    )
                            # Draw a cube with 12 axes
                            line_pairs = [
                                [0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 0],
                                [4, 5],
                                [5, 6],
                                [6, 7],
                                [7, 4],
                                [0, 4],
                                [1, 5],
                                [2, 6],
                                [3, 7],
                            ]
                            for start, end in line_pairs:
                                cv2.line(debug_frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                    except:
                        pass
            if camera:
                cv2.imshow("Camera view", debug_frame)

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        pass

fps.stop()
print("FPS: {:.2f}".format(fps.fps()))
