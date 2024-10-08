import numpy as np
import depthai as dai
from utils import VehiclePersonDetection

device = dai.Device()
platform = device.getPlatform()
nn_input_shape = (256,192)
modelDescription = dai.NNModelDescription(
    modelSlug="thermalpersonvehicledetection",
    modelVersionSlug="personvehicle",
    platform=platform.name,
    modelInstanceSlug=f"{nn_input_shape[1]}x{nn_input_shape[0]}",

)
archivePath = dai.getModelFromZoo(modelDescription, apiKey="API_KEY", useCached=True) # Replace API_KEY with your API key
nn_archive = dai.NNArchive(archivePath)


with dai.Pipeline(device) as pipeline:
    # Change color camera to thermal camera. Note that thermal.color is YUV422i and we need it to be 3D grayscale image
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewSize(nn_input_shape)
    cam.setInterleaved(False)

    # Play with setIouThreshold() and setConfidenceThreshold() to get better results
    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setNNArchive(nn_archive)

    output = cam.preview
    cam.preview.link(nn.input)

    pipeline.create(VehiclePersonDetection).build(
        img_frame=output, 
        detections=nn.out,
        )
    
    print("Pipeline created")
    pipeline.run()
    print("Pipeline finished")
