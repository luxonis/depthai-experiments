import depthai as dai
from host_palm_detection import PalmDetection

modelDescription = dai.NNModelDescription(modelSlug="mediapipe-palm-detection", platform="RVC2")
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(40)
    cam.setIspScale(2, 3) # 720P
    cam.setVideoSize(720, 720)
    cam.setPreviewSize(128, 128)
    cam.setInterleaved(False)

    nn_archive = dai.NNArchive(archivePath)
    model_nn = pipeline.create(dai.node.NeuralNetwork)
    model_nn.setNNArchive(nn_archive)
    
    model_nn.input.setBlocking(False)
    cam.preview.link(model_nn.input)

    palm_detection = pipeline.create(PalmDetection).build(
        preview=cam.video,
        palm_detections=model_nn.out
    )

    print("Pipeline created.")
    pipeline.run()

