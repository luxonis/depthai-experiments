import blobconverter
import depthai as dai
import re
from host_triangulation import Triangulation

faceDet_modelDescription = dai.NNModelDescription(modelSlug="yunet", platform="RVC2")
faceDet_archivePath = dai.getModelFromZoo(faceDet_modelDescription)
faceDet_nnarchive = dai.NNArchive(faceDet_archivePath)

# Creates and connects nodes, once for the left camera and once for the right camera
def populate_pipeline(p: dai.Pipeline, left: bool, resolution: dai.MonoCameraProperties.SensorResolution)\
        -> tuple[dai.Node.Output, dai.Node.Output, dai.Node.Output]:
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.CAM_B if left else dai.CameraBoardSocket.CAM_C
    cam.setBoardSocket(socket)
    cam.setResolution(resolution)

    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(640, 640)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    face_manip.setMaxOutputFrameSize(640*640*3)
    cam.out.link(face_manip.inputImage)

    face_nn = p.create(dai.node.MobileNetDetectionNetwork)
    face_nn.setConfidenceThreshold(0.2)
    face_nn.setBlobPath(blobconverter.from_zoo("face-detection-retail-0004", shaves=6, version="2021.4"))

    face_manip.out.link(face_nn.input)

    # Script node will take the output from the NN as an input, get the first bounding box
    # and send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    image_manip_script.inputs['nn_in'].setBlocking(False)
    image_manip_script.inputs['nn_in'].setMaxSize(1)
    face_nn.out.link(image_manip_script.inputs['nn_in'])
    image_manip_script.setScript("""
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    face_dets = node.io['nn_in'].get().detections
    if len(face_dets) == 0: continue
    coords = face_dets[0] # take first
    
    coords.xmin -= 0.05
    coords.ymin -= 0.05
    coords.xmax += 0.05
    coords.ymax += 0.05
    
    limit_roi(coords)
    cfg = ImageManipConfig()
    cfg.setKeepAspectRatio(False)
    cfg.setCropRect(coords.xmin, coords.ymin, coords.xmax, coords.ymax)
    cfg.setResize(48, 48)
    node.io['to_manip'].send(cfg)
""")

    manip_crop = p.create(dai.node.ImageManip)
    face_nn.passthrough.link(manip_crop.inputImage)
    image_manip_script.outputs['to_manip'].link(manip_crop.inputConfig)
    manip_crop.initialConfig.setResize(48, 48)
    manip_crop.inputConfig.setWaitForMessage(False)

    landmarks_nn = p.create(dai.node.NeuralNetwork)
    landmarks_nn.setBlobPath(blobconverter.from_zoo("landmarks-regression-retail-0009", shaves=6, version="2021.4"))
    manip_crop.out.link(landmarks_nn.input)

    return (face_manip.out, face_nn.out, landmarks_nn.out)

device = dai.Device()
with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

    face_left, face_nn_left, landmarks_nn_left = populate_pipeline(pipeline, True, resolution)
    face_right, face_nn_right, landmarks_nn_right = populate_pipeline(pipeline, False, resolution)

    triangulation = pipeline.create(Triangulation).build(
        face_left=face_left,
        face_right=face_right,
        face_nn_left=face_nn_left,
        face_nn_right=face_nn_right,
        landmarks_nn_left=landmarks_nn_left,
        landmarks_nn_right=landmarks_nn_right,
        device=device,
        # THE_720_P => 720
        resolution_number=int(re.findall(r"\d+", str(resolution))[0])
    )

    print("Pipeline created.")
    pipeline.run()
