import depthai as dai
from depthai_nodes import MPPalmDetectionParser


device = dai.Device()
platform = device.getPlatform()
model_dimension = 192
modelDescription = dai.NNModelDescription(
    modelSlug="mediapipe-palm-detection",
    platform=platform.name,
    modelVersionSlug=f"{model_dimension}x{model_dimension}",
)
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)

visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output_type = dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p
    cam_nn = cam.requestOutput((model_dimension, model_dimension), output_type)
    cam_output = cam.requestOutput((model_dimension * 5, model_dimension * 5), dai.ImgFrame.Type.NV12)
    
    nn_archive = dai.NNArchive(archivePath)
    model_nn = pipeline.create(dai.node.NeuralNetwork).build(cam_nn, nn_archive)
    model_nn.input.setBlocking(False)

    parser = pipeline.create(MPPalmDetectionParser)
    parser.setScale(192)
    model_nn.out.link(parser.input)

    visualizer.addTopic("Palm Detection", cam_output)
    visualizer.addTopic("Palm Bounding Boxes", parser.out)
    #visualizer.registerPipeline(pipeline) # TODO: remove comment when registerPipeline is fixed
    
    print("Pipeline created.")
    pipeline.run()
