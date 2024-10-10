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
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(512, 288)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(40)

    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
        input=camRgb.preview, nnArchive=nn_archive
    )

    pipeline.create(DeviceDecoding).build(
        images=camRgb.preview, detections=detectionNetwork.out
    )

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")
