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
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


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
    print("Creating Person Reidentification Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(str(Path("models/person-reidentification-retail-0031.blob").resolve().absolute()))
    reid_nn_xout = pipeline.createXLinkOut()
    reid_nn_xout.setStreamName("reid_nn")
    reid_in.out.link(reid_nn.input)
    reid_nn.out.link(reid_nn_xout.input)

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
    reid_in = device.getInputQueue("reid_in")
    reid_nn = device.getOutputQueue("reid_nn")

    bboxes = []
    results = {}
    results_path = {}
    reid_bbox_q = queue.Queue()
    next_id = 0

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
                    reid_in.send(nn_data)
                    reid_bbox_q.put(bbox)

            while reid_nn.has():
                reid_result = reid_nn.get().getFirstLayerFp16()
                bbox = reid_bbox_q.get()

                for person_id in results:
                    dist = cos_dist(reid_result, results[person_id])
                    if dist > 0.7:
                        result_id = person_id
                        results[person_id] = reid_result
                        break
                else:
                    result_id = next_id
                    results[result_id] = reid_result
                    results_path[result_id] = []
                    next_id += 1

                if debug:
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    x = (bbox[0] + bbox[2]) // 2
                    y = (bbox[1] + bbox[3]) // 2
                    results_path[result_id].append([x, y])
                    cv2.putText(debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                    if len(results_path[result_id]) > 1:
                        cv2.polylines(debug_frame, [np.array(results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)
                else:
                    print(f"Saw id: {result_id}")

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
