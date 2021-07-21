import argparse
import queue
import time
from pathlib import Path
import blobconverter
import cv2
import depthai
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


def cos_dist(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if args.camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(300, 300)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    detection_nn = pipeline.createMobileNetDetectionNetwork()
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setBlobPath(str(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6 if args.camera else 8
    )))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)

    if args.camera:
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    # NeuralNetwork
    print("Creating Age Gender Neural Network...")
    age_gender_in = pipeline.createXLinkIn()
    age_gender_in.setStreamName("age_gender_in")
    age_gender_nn = pipeline.createNeuralNetwork()
    age_gender_nn.setBlobPath(str(blobconverter.from_zoo(
        name="age-gender-recognition-retail-0013",
        shaves=6 if args.camera else 8
    )))
    age_gender_nn_xout = pipeline.createXLinkOut()
    age_gender_nn_xout.setStreamName("age_gender_nn")
    age_gender_in.out.link(age_gender_nn.input)
    age_gender_nn.out.link(age_gender_nn_xout.input)

    print("Pipeline created.")
    return pipeline


with depthai.Device() as device:
    print("Starting pipeline...")
    device.startPipeline(create_pipeline())
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        detection_in = device.getInputQueue("detection_in")
    detection_nn = device.getOutputQueue("detection_nn")
    age_gender_in = device.getInputQueue("age_gender_in")
    age_gender_nn = device.getOutputQueue("age_gender_nn")

    detections = []
    results = []
    face_bbox_q = queue.Queue()
    next_id = 0

    if args.video:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))

    fps = FPS()
    fps.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if args.video:
            return cap.read()
        else:
            return True, np.array(cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)


    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            if frame is not None:
                fps.update()
                debug_frame = frame.copy()

                if not args.camera:
                    nn_data = depthai.NNData()
                    nn_data.setLayer("input", to_planar(frame, (300, 300)))
                    detection_in.send(nn_data)

            while detection_nn.has():
                detections = detection_nn.get().detections
                for detection in detections:
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    age_gender_in.send(nn_data)
                    face_bbox_q.put(bbox)

            while age_gender_nn.has():
                det = age_gender_nn.get()
                age = int(float(np.squeeze(np.array(det.getLayerFp16('age_conv3')))) * 100)
                gender = np.squeeze(np.array(det.getLayerFp16('prob')))
                gender_str = "female" if gender[0] > gender[1] else "male"
                bbox = face_bbox_q.get()

                while not len(results) < len(detections) and len(results) > 0:
                    results.pop(0)
                results.append({
                    "bbox": bbox,
                    "gender": gender_str,
                    "age": age,
                    "ts": time.time()
                })

            results = list(filter(lambda result: time.time() - result["ts"] < 0.2, results))

            if debug and frame is not None:
                for result in results:
                    bbox = result["bbox"]
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    y = (bbox[1] + bbox[3]) // 2
                    cv2.putText(debug_frame, str(result["age"]), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                    cv2.putText(debug_frame, result["gender"], (bbox[0], y + 20), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))

                aspect_ratio = frame.shape[1] / frame.shape[0]
                cv2.imshow("Camera_view", cv2.resize(debug_frame, (int(900),  int(900 / aspect_ratio))))
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        pass

fps.stop()
print("FPS: {:.2f}".format(fps.fps()))
if args.video:
    cap.release()
