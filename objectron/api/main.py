import blobconverter
import cv2
import depthai as dai
import numpy as np

from depthai_sdk.fps import FPSHandler
from functions import draw_box

# ---------- Parameters ----------

MOBILENET_DETECTOR_PATH = str(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
OBJECTRON_CHAIR_PATH = "models/objectron_chair_openvino_2021.4_6shave.blob"

INPUT_W, INPUT_H = 640, 360


# ---------- Pipeline ----------

# Camera --> Manip --> Detection NN --> ScriptNode --> Manip --> Pose NN --> Result

def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

    # Create a camera
    cam = pipeline.createColorCamera()
    cam.setIspScale(1, 3)
    cam.initialControl.setManualFocus(130)
    cam.setPreviewSize(INPUT_W, INPUT_H)
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setPreviewKeepAspectRatio(True)
    cam.setFps(30)

    # Create a Manip that resizes image before detection
    manip_det = pipeline.create(dai.node.ImageManip)
    manip_det.initialConfig.setResize(300, 300)
    manip_det.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

    # Mobilenet parameters
    nn_det = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn_det.setBlobPath(MOBILENET_DETECTOR_PATH)
    nn_det.setConfidenceThreshold(0.5)
    nn_det.setNumInferenceThreads(2)

    # Connect Camera to Manip
    cam.preview.link(manip_det.inputImage)

    # Out links
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam")
    cam.preview.link(cam_xout.input)

    nn_det_xout = pipeline.createXLinkOut()
    nn_det_xout.setStreamName("nn_det")
    nn_det.out.link(nn_det_xout.input)

    manip_det.out.link(nn_det.input)

    # ScriptNode - take the output of the detector, postprocess, and send info to ImageManip for resizing
    # the input to the pose NN
    script_pp = pipeline.create(dai.node.Script)
    nn_det.out.link(script_pp.inputs["det_in"])
    nn_det.passthrough.link(script_pp.inputs['det_passthrough'])

    f = open("script.py", "r")
    script_txt = f.read()
    script_pp.setScript(script_txt)

    script_pp.setProcessor(dai.ProcessorType.LEON_CSS)

    # Connect camera preview to scripts input to use full image size
    cam.preview.link(script_pp.inputs['preview'])

    # Create Manip for frame resizing before pose NN
    manip_pose = pipeline.createImageManip()
    manip_pose.initialConfig.setResize(224, 224)
    manip_pose.inputConfig.setWaitForMessage(True)
    manip_pose.inputConfig.setQueueSize(10)
    manip_pose.inputImage.setQueueSize(10)

    # Link scripts output to pose Manip
    script_pp.outputs['manip_cfg'].link(manip_pose.inputConfig)
    script_pp.outputs['manip_img'].link(manip_pose.inputImage)

    # Link manip config to queue
    config_xout = pipeline.createXLinkOut()
    config_xout.setStreamName("cfg_out")
    script_pp.outputs['manip_cfg'].link(config_xout.input)

    # 2nd stage pose NN
    nn_pose = pipeline.createNeuralNetwork()
    nn_pose.setBlobPath(OBJECTRON_CHAIR_PATH)
    manip_pose.out.link(nn_pose.input)
    nn_pose.input.setQueueSize(20)

    nn_pose_xout = pipeline.createXLinkOut()
    nn_pose_xout.setStreamName("nn_pose")
    nn_pose.out.link(nn_pose_xout.input)

    nn_pose_frame_xout = pipeline.createXLinkOut()
    nn_pose_frame_xout.setStreamName("nn_pose_frame")
    nn_pose.passthrough.link(nn_pose_frame_xout.input)

    return pipeline


# --------------- Main ---------------

if __name__ == "__main__":

    frame_seq_map = {}
    pose_map = {}

    print("Starting pipeline...")
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:

        q_cam = device.getOutputQueue("cam", 4, False)
        q_detection = device.getOutputQueue("nn_det", 4, False)

        q_pose = device.getOutputQueue("nn_pose", 20, False)
        q_pose_frame = device.getOutputQueue("nn_pose_frame", 16, False)
        q_cfg = device.getOutputQueue("cfg_out", 16, False)

        fps = FPSHandler(maxTicks=2)

        frame = None

        while True:

            in_frame = q_cam.tryGet()
            if in_frame is None:
                continue

            frame = in_frame.getCvFrame()

            frame_seq_map[in_frame.getSequenceNum()] = frame.copy()

            in_det = q_detection.tryGet()

            fps.tick("tock")

            if in_det is not None:
                detections = in_det.detections
                # find the detection with highest confidence
                # label of 9 indicates chair in VOC
                confs = [det.confidence for det in detections if det.label == 9]
                # confs = [det.confidence for det in detections if det.label == 62]
                if len(confs) > 0:
                    idx = confs.index(max(confs))
                    detection = detections[idx]

                    # try and get pose_frame
                    # if pose_frame exists, so does the config

                    pose_frame_in = q_pose_frame.tryGet()
                    if pose_frame_in is not None:

                        # POSE INFERENCE
                        in_pose = q_pose.get()
                        in_cfg = q_cfg.get()

                        out = np.array(in_pose.getLayerFp16("StatefulPartitionedCall:1")).reshape(9, 2)

                        cfg_crop = in_cfg

                        pose_frame = pose_frame_in.getCvFrame()

                        # if out is not None and cfg_crop is not None:
                        xmin, ymin = cfg_crop.getCropXMin(), cfg_crop.getCropYMin()
                        xmax, ymax = cfg_crop.getCropXMax(), cfg_crop.getCropYMax()

                        xmin, ymin = int(xmin * INPUT_W), int(ymin * INPUT_H)
                        xmax, ymax = int(xmax * INPUT_W), int(ymax * INPUT_H)

                        OG_W, OG_H = xmax - xmin, ymax - ymin

                        scale_x = OG_W / 224
                        scale_y = OG_H / 224

                        out[:, 0] = out[:, 0] * scale_x + xmin
                        out[:, 1] = out[:, 1] * scale_y + ymin

                        l = pose_map.get(pose_frame_in.getSequenceNum())
                        if l is None:
                            l = []
                        l.append(out)
                        pose_map[pose_frame_in.getSequenceNum()] = l

            cv2.putText(frame, "NN fps: {:.2f}".format(fps.tickFps("tock")), (2, frame.shape[0] - 4),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        (255, 0, 0))

            # show delayed input with synced output if exists

            frame_synced = img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)

            if len(frame_seq_map) > 30:
                seq_id = list(frame_seq_map.keys())[0]

                frame_synced = frame_seq_map[seq_id]
                boxes = pose_map.get(seq_id)
                if boxes is not None:
                    for box in boxes:
                        draw_box(frame_synced, box)

                for map_key in list(filter(lambda item: item <= seq_id, frame_seq_map.keys())):
                    del frame_seq_map[map_key]
                for map_key in list(filter(lambda item: item <= seq_id, pose_map.keys())):
                    del pose_map[map_key]

            frame = cv2.hconcat([frame, frame_synced])

            cv2.imshow("Objectron", frame)

            if cv2.waitKey(1) == ord('q'):
                break
