import depthai as dai
from device_decoding import DeviceDecoding

device = dai.Device()

modelDescription = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)


with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    color_out = cam.requestOutput(
        size=(512, 288), type=dai.ImgFrame.Type.BGR888p, fps=40
    )

    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
        input=color_out, nnArchive=nn_archive
    )

    pipeline.create(DeviceDecoding).build(
        images=color_out, detections=detectionNetwork.out
    )

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")
