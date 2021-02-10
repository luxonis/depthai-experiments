import argparse
import queue
from pathlib import Path

import cv2
import depthai
print(depthai.__version__)
import numpy as np
from imutils.video import FPS

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug
camera = not args.video

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")


def cos_dist(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }



def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(544, 320)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)
    if camera:
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(
        str(Path("models/face-detection-retail-0004.blob").resolve().absolute()))
    face_in = pipeline.createXLinkIn()
    face_in.setStreamName("face_in")
    face_in.out.link(face_nn.input)
    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    # NeuralNetwork
    print("Creating Landmarks Detection Neural Network...")
    land_nn = pipeline.createNeuralNetwork()
    land_nn.setBlobPath(
        str(Path("models/landmarks-regression-retail-0009.blob").resolve().absolute())
    )
    land_nn_xin = pipeline.createXLinkIn()
    land_nn_xin.setStreamName("landmark_in")
    land_nn_xin.out.link(land_nn.input)
    land_nn_xout = pipeline.createXLinkOut()
    land_nn_xout.setStreamName("landmark_nn")
    land_nn.out.link(land_nn_xout.input)

    print("Pipeline created.")
    return pipeline


with depthai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    device.startPipeline()
    if camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        cap = cv2.VideoCapture(str(Path("./input.mp4").resolve().absolute()))
        detection_in = device.getInputQueue("detection_in")
    detection_nn = device.getOutputQueue("detection_nn")
    face_in = device.getInputQueue("face_in")
    face_nn = device.getOutputQueue("face_nn")
    land_in = device.getOutputQueue("landmark_in")
    land_nn = device.getOutputQueue("landmark_nn")

    bboxes = []
    face_box_q = queue.Queue()

    fps = FPS()
    fps.start()


    def should_run():
        return True if camera else cap.isOpened()


    def get_frame():
        if camera:
            return True, np.array(cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)
        else:
            return cap.read()

    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            if frame is not None:
                fps.update()
                debug_frame = frame.copy()

                if not camera:
                    nn_data = depthai.NNData()
                    nn_data.setLayer("input", to_planar(frame, (544, 320)))
                    detection_in.send(nn_data)

            while detection_nn.has():
                bboxes = np.array(detection_nn.get().getFirstLayerFp16())
                bboxes = bboxes.reshape((bboxes.size // 7, 7))
                bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

                for raw_bbox in bboxes:
                    bbox = frame_norm(frame, raw_bbox)
                    det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    face_in.send(nn_data)


            def face_thread():
                face_nn = device.getOutputQueue("face_nn")
                landmark_in = device.getInputQueue("landmark_in")

                while should_run():
                    if frame is None:
                        continue
                    try:
                        bboxes = np.array(face_nn.get().getFirstLayerFp16())
                    except RuntimeError as ex:
                        continue
                    bboxes = bboxes.reshape((bboxes.size // 7, 7))
                    bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

                    for raw_bbox in bboxes:
                        bbox = frame_norm(frame, raw_bbox)
                        det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                        land_data = depthai.NNData()
                        land_data.setLayer("0", to_planar(det_frame, (48, 48)))
                        landmark_in.send(land_data)

                        face_box_q.put(bbox)


            def land_pose_thread():
                landmark_nn = device.getOutputQueue(name="landmark_nn", maxSize=1, blocking=False)

                while should_run():
                    try:
                        land_in = landmark_nn.get().getFirstLayerFp16()
                    except RuntimeError as ex:
                        continue

                    try:
                        face_bbox = face_box_q.get(block=True, timeout=100)
                    except queue.Empty:
                        continue

                    face_box_q.task_done()
                    left = face_bbox[0]
                    top = face_bbox[1]
                    face_frame = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                    land_data = frame_norm(face_frame, land_in)
                    land_data[::2] += left
                    land_data[1::2] += top
                    nose = land_data[4:6]

            if debug:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                cv2.imshow("Camera_view", cv2.resize(debug_frame, (int(900),  int(900 / aspect_ratio))))
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        pass

    fps.stop()
    print("FPS: {:.2f}".format(fps.fps()))
    if not camera:
        cap.release()
