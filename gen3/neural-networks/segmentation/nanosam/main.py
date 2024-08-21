import depthai as dai

from host_nanosam_main import NanoSAM

modelDescription = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2", modelVersionSlug="coco-416x416") 
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive_det = dai.NNArchive(archivePath)

modelDescription = dai.NNModelDescription(modelSlug="nanosam-resnet18-image-encoder", platform="RVC2", modelVersionSlug="1024x1024")
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive_enc = dai.NNArchive(archivePath)

detection_shape = (416, 416)
nn_shape = (1024, 1024)

with dai.Pipeline() as pipeline:

    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb_previem = cam.requestOutput(size=nn_shape, type=dai.ImgFrame.Type.BGR888p) 

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*detection_shape)
    manip.inputConfig.setWaitForMessage(False)
    rgb_previem.link(manip.inputImage)

    detection_nn = pipeline.create(dai.node.DetectionNetwork).build(input=manip.out, nnArchive=nn_archive_det, confidenceThreshold=0.5)
    detection_nn.setNumInferenceThreads(2)

    embedding_nn = pipeline.create(dai.node.NeuralNetwork)
    embedding_nn.setNNArchive(nn_archive_enc)
    embedding_nn.setNumInferenceThreads(2)
    rgb_previem.link(embedding_nn.input)

    nanosam = pipeline.create(NanoSAM).build(
        preview=rgb_previem,
        detections=detection_nn.out,
        nn=embedding_nn.out
    )
    nanosam.inputs["preview"].setBlocking(False)
    nanosam.inputs["preview"].setMaxSize(3)
    nanosam.inputs["detections"].setBlocking(False)
    nanosam.inputs["detections"].setMaxSize(3)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")