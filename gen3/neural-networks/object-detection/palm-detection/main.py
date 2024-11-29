import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
import time

IP = "10.12.121.85"
deviceInfo = dai.DeviceInfo(IP)
device = dai.Device(deviceInfo)
# device = dai.Device()
platform = device.getPlatform()

visualizer = dai.RemoteConnection(webSocketPort=8765, httpPort=8082)
ENCODER_PROFILE = dai.VideoEncoderProperties.Profile.MJPEG

description1 = dai.NNModelDescription("luxonis/yunet:new-480x640:0.0.1")
dai.ImgDetection
### 
# need a better way to get model shape for the output, other option is to use networkNode.passthrough
description = dai.NNModelDescription("luxonis/yunet:new-480x640:0.0.1", 
                                     platform="RVC4")
archive_path = dai.getModelFromZoo(description, useCached=False)
archive = dai.NNArchive(archivePath=archive_path)
model_shape =  archive.getConfig().model.inputs[0].shape[1:3]
model_shape = (model_shape[1]*2, model_shape[0]*2)
###
dai.NNData
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    networkNode = pipeline.create(ParsingNeuralNetwork).build(
        cameraNode, description1, fps=15 # FPS would need to be adjusted based on the model + RVC version
        )
    
    visualizer_output = cameraNode.requestOutput(model_shape, dai.ImgFrame.Type.NV12, fps=15)
    visualizer.addTopic("Video", visualizer_output, "images") 
    visualizer.addTopic("Visualizations", networkNode.out, "images")
    
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        time.sleep(1/30)
