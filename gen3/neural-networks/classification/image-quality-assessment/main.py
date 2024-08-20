import depthai as dai
from host_image_quality_assessment import ImageQualityAssessment

device = dai.Device()
model_description = dai.NNModelDescription(modelSlug="image-quality-assessment", platform=device.getPlatform().name)
nn_path = dai.getModelFromZoo(modelDescription=model_description)

nn_archive = dai.NNArchive(nn_path)

with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb_preview = cam.requestOutput(size=(256, 256), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    rgb_preview.link(nn.input)

    quality_assessment = pipeline.create(ImageQualityAssessment).build(
        preview=rgb_preview,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
