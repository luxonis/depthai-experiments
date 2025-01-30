import depthai as dai
import blobconverter

from host_fatigue_detection import FatigueDetection

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setIspScale(2, 3)
    cam.setInterleaved(False)
    cam.setPreviewSize(720, 720)

    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    cam.preview.link(face_det_manip.inputImage)

    face_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_nn.setConfidenceThreshold(0.3)
    face_nn.setBlobPath(blobconverter.from_zoo("face-detection-retail-0004", shaves=6))
    face_det_manip.out.link(face_nn.input)

    # Script node will take the output from the NN as an input, get the first bounding box
    # and send ImageManipConfig to the manip_crop
    script = pipeline.create(dai.node.Script)
    script.inputs["nn_in"].setBlocking(False)
    script.inputs["nn_in"].setMaxSize(1)
    face_nn.out.link(script.inputs["nn_in"])
    script.setScript("""
import time
def enlrage_roi(det): # For better face landmarks NN results
    det.xmin -= 0.05
    det.ymin -= 0.02
    det.xmax += 0.05
    det.ymax += 0.02
    
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    nn_out = node.io['nn_in'].tryGet()

    if nn_out is not None:
        face_dets = nn_out.detections
    
        # No faces found
        if len(face_dets) == 0: continue
    
        det = face_dets[0] # Take first
        enlrage_roi(det)
        limit_roi(det)
    
        cfg = ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        cfg.setResize(160, 160)
        cfg.setKeepAspectRatio(False)
        node.io['manip_cfg'].send(cfg)
    """)

    crop_manip = pipeline.create(dai.node.ImageManip)
    script.outputs["manip_cfg"].link(crop_manip.inputConfig)
    face_det_manip.out.link(crop_manip.inputImage)
    crop_manip.initialConfig.setResize(160, 160)
    crop_manip.inputConfig.setWaitForMessage(False)

    landmarks_nn = pipeline.create(dai.node.NeuralNetwork)
    landmarks_nn.setBlobPath(
        blobconverter.from_zoo(
            name="facial_landmarks_68_160x160",
            shaves=6,
            zoo_type="depthai",
            version="2021.4",
        )
    )
    crop_manip.out.link(landmarks_nn.input)

    fatigue_detection = pipeline.create(FatigueDetection).build(
        preview=cam.preview, face_nn=face_nn.out, landmarks_nn=landmarks_nn.out
    )
    fatigue_detection.inputs["preview"].setBlocking(False)
    fatigue_detection.inputs["preview"].setMaxSize(1)
    fatigue_detection.inputs["face_dets"].setBlocking(False)
    fatigue_detection.inputs["face_dets"].setMaxSize(1)
    fatigue_detection.inputs["landmarks"].setBlocking(False)
    fatigue_detection.inputs["landmarks"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
