import depthai as dai
from host_social_distancing import SocialDistancing

device = dai.Device()
modelDescription = dai.NNModelDescription(modelSlug="person-detection-retail", platform=device.getPlatform().name, modelVersionSlug="0013-544x320")
# modelDescription = dai.NNModelDescription(modelSlug="scrfd-person-detection", platform=device.getPlatform().name, modelVersionSlug="2-5g-640x640") # no parser yet
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)
nnArchive = dai.NNArchive(archivePath)

with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb = cam.requestOutput(size=(544, 320), type=dai.ImgFrame.Type.BGR888p, fps=20)

    left = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.requestOutput((640, 400)), right=right.requestOutput((640, 400)))
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(544, 320)

    nn = pipeline.create(dai.node.SpatialDetectionNetwork)
    nn.setNNArchive(nnArchive)

    nn.input.setBlocking(False)
    nn.setConfidenceThreshold(0.5)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100)
    nn.setDepthUpperThreshold(5000)
    
    rgb.link(nn.input)
    stereo.depth.link(nn.inputDepth)

    social_distancing = pipeline.create(SocialDistancing).build(
        preview=rgb,
        nn=nn.out
    )
    social_distancing.inputs["preview"].setBlocking(False)
    social_distancing.inputs["preview"].setMaxSize(4)
    social_distancing.inputs["detections"].setBlocking(False)
    social_distancing.inputs["detections"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline exited.")