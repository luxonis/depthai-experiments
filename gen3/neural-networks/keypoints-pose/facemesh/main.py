import depthai as dai
import blobconverter
from host_facemesh import Facemesh

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setPreviewKeepAspectRatio(True)
    cam.setPreviewSize(540, 360)
    cam.setIspScale(1, 3)
    cam.setVideoSize(1080, 720)
    cam.setInterleaved(False)

    manip_to_nn = pipeline.create(dai.node.ImageManip)
    manip_to_nn.initialConfig.setResize(300, 300)
    manip_to_nn.initialConfig.setKeepAspectRatio(True)
    cam.preview.link(manip_to_nn.inputImage)

    face_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    face_nn.setConfidenceThreshold(0.5)
    face_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    manip_to_nn.out.link(face_nn.input)

    script = pipeline.create(dai.node.Script)
    face_nn.out.link(script.inputs['dets'])
    script.setScript(f"""
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

FACE_BBOX_PADDING = 0.06
while True:
    face_dets = node.io['dets'].get().detections
    if len(face_dets) == 0: continue
    coords = face_dets[0] # take first
    
    coords.xmin -= FACE_BBOX_PADDING
    coords.ymin -= FACE_BBOX_PADDING
    coords.xmax += FACE_BBOX_PADDING
    coords.ymax += FACE_BBOX_PADDING
    
    limit_roi(coords)
    cfg = ImageManipConfig()
    cfg.setKeepAspectRatio(False)
    cfg.setCropRect(coords.xmin, coords.ymin, coords.xmax, coords.ymax)
    cfg.setResize(192, 192)
    node.io['cfg'].send(cfg)
""")

    crop_face = pipeline.create(dai.node.ImageManip)
    crop_face.setMaxOutputFrameSize(3110400)
    crop_face.inputConfig.setWaitForMessage(False)
    crop_face.initialConfig.setResize(192, 192)
    script.outputs['cfg'].link(crop_face.inputConfig)
    manip_to_nn.out.link(crop_face.inputImage)

    landmarks_nn = pipeline.create(dai.node.NeuralNetwork)
    landmarks_nn.setBlobPath("face_landmark_openvino_2021.4_6shave.blob")
    landmarks_nn.setNumPoolFrames(4)
    landmarks_nn.input.setBlocking(False)
    landmarks_nn.setNumInferenceThreads(2)
    crop_face.out.link(landmarks_nn.input)

    facemesh = pipeline.create(Facemesh).build(
        full=cam.preview,
        preview=manip_to_nn.out,
        face_nn=face_nn.out,
        landmarks_nn=landmarks_nn.out
    )

    print("Pipeline created.")
    pipeline.run()
