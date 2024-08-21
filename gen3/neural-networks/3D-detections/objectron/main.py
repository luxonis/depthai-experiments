import cv2
import depthai as dai
from functions import draw_box
import numpy as np
from utility.fps_handler import FPSHandler

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2", modelVersionSlug="300x300")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

OBJECTRON_CHAIR_PATH = "models/objectron_chair_openvino_2021.4_6shave.blob"
INPUT_W, INPUT_H = 427, 267


class DisplayChairDetections(dai.node.HostNode):
    def __init__(self):
        self._fps = FPSHandler()
        super().__init__()


    def build(self, cam_out : dai.Node.Output, det_out : dai.Node.Output, pose_out : dai.Node.Output,
              pose_frame_out : dai.Node.Output, cfg_out : dai.Node.Output) -> "DisplayChairDetections":
        
        self.frame_seq_map = {}
        self.pose_map = {}

        self.q_cfg = cfg_out.createOutputQueue(16, False)
        self.q_pose = pose_out.createOutputQueue(20, False)
        self.q_pose_frame = pose_frame_out.createOutputQueue(16, False)

        self.link_args(cam_out, det_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.ImgFrame, in_det : dai.ImgDetections) -> None:

        if in_frame is not None:
            self._fps.next_iter()

            frame = in_frame.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(self._fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))

            self.frame_seq_map[in_frame.getSequenceNum()] = frame.copy()

            if in_det is not None:
                detections = in_det.detections
                # find the detection with highest confidence
                # label of 9 indicates chair in VOC
                confs = [det.confidence for det in detections if det.label == 9]
                # confs = [det.confidence for det in detections if det.label == 62]
                if len(confs) > 0:
                    # try and get pose_frame
                    # if pose_frame exists, so does the config

                    pose_frame_in : dai.ImgFrame = self.q_pose_frame.tryGet()
                    if pose_frame_in is not None:

                        # POSE INFERENCE
                        in_pose : dai.NNData = self.q_pose.get()
                        in_cfg : dai.ImageManipConfig = self.q_cfg.get()

                        out = np.array(in_pose.getTensor("StatefulPartitionedCall:1")).reshape(9, 2)

                        cfg_crop = in_cfg

                        #if out is not None and cfg_crop is not None:
                        xmin, ymin = cfg_crop.getCropXMin(), cfg_crop.getCropYMin()
                        xmax, ymax = cfg_crop.getCropXMax(), cfg_crop.getCropYMax()

                        xmin, ymin = int(xmin * INPUT_W), int(ymin * INPUT_H)
                        xmax, ymax = int(xmax * INPUT_W), int(ymax * INPUT_H)

                        OG_W, OG_H = xmax - xmin, ymax - ymin

                        scale_x = OG_W / 224
                        scale_y = OG_H / 224

                        out[:, 0] = out[:, 0] * scale_x + xmin
                        out[:, 1] = out[:, 1] * scale_y + ymin

                        l = self.pose_map.get(pose_frame_in.getSequenceNum())
                        if l is None:
                            l = []
                        l.append(out)
                        self.pose_map[pose_frame_in.getSequenceNum()] = l

            # show delayed input with synced output if exists

            frame_synced = np.zeros((INPUT_H, INPUT_W,3), dtype=np.uint8)

            if len(self.frame_seq_map) > 30:
                seq_id = list(self.frame_seq_map.keys())[0]

                frame_synced = self.frame_seq_map[seq_id]
                boxes = self.pose_map.get(seq_id)
                if boxes is not None:
                    for box in boxes:
                        draw_box(frame_synced, box)

                for map_key in list(filter(lambda item: item <= seq_id, self.frame_seq_map.keys())):
                    del self.frame_seq_map[map_key]
                for map_key in list(filter(lambda item: item <= seq_id, self.pose_map.keys())):
                    del self.pose_map[map_key]

            frame = cv2.hconcat([frame, frame_synced])

            cv2.imshow("Objectron", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


with dai.Pipeline() as pipeline:

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # Create a camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setIspScale(1, 3)
    cam.initialControl.setManualFocus(130)
    cam.setPreviewSize(INPUT_W, INPUT_H)
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewKeepAspectRatio(True)
    cam.setFps(60)

    # Create a Manip that resizes image before detection
    manip_det = pipeline.create(dai.node.ImageManip)
    manip_det.initialConfig.setResize(300, 300)
    manip_det.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

    # Mobilenet parameters
    nn_det = pipeline.create(dai.node.DetectionNetwork)
    nn_det.setNNArchive(nn_archive)
    nn_det.setConfidenceThreshold(0.5)
    nn_det.setNumInferenceThreads(2)

    # Connect Camera to Manip
    cam.preview.link(manip_det.inputImage)
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
    # manip_pose = pipeline.createImageManip()
    manip_pose = pipeline.create(dai.node.ImageManip)
    manip_pose.initialConfig.setResize(224, 224)
    manip_pose.inputConfig.setWaitForMessage(True)
    manip_pose.inputConfig.setMaxSize(10)
    manip_pose.inputImage.setMaxSize(10)

    # Link scripts output to pose Manip
    script_pp.outputs['manip_cfg'].link(manip_pose.inputConfig)
    script_pp.outputs['manip_img'].link(manip_pose.inputImage)

    # 2nd stage pose NN
    nn_pose = pipeline.create(dai.node.NeuralNetwork)
    nn_pose.setBlobPath(OBJECTRON_CHAIR_PATH)
    manip_pose.out.link(nn_pose.input)
    nn_pose.input.setMaxSize(20)

    detector = pipeline.create(DisplayChairDetections).build(
        cam_out=cam.preview,
        det_out=nn_det.out,
        pose_out=nn_pose.out,
        pose_frame_out=nn_pose.passthrough,
        cfg_out=script_pp.outputs['manip_cfg']        
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
