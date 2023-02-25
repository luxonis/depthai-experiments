import cv2
import depthai as dai
import numpy as np
import blobconverter
from skimage.segmentation import mark_boundaries

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(1080,1080)
camRgb.setInterleaved(False)

# 1080x1080 -> 256x256 required by the model
scale_manip = pipeline.create(dai.node.ImageManip)
scale_manip.initialConfig.setResize(256,256)
camRgb.preview.link(scale_manip.inputImage)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blobconverter.from_zoo(name="padim_wood_256x256", zoo_type="depthai", shaves=6))
scale_manip.out.link(nn.input)

# Linking
frameOut = pipeline.create(dai.node.XLinkOut)
frameOut.setStreamName("color")
nn.passthrough.link(frameOut.input)
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to a device and start the pipeline
with dai.Device(pipeline) as device:

    qColor = device.getOutputQueue("color", maxSize=4, blocking=False)
    qNN = device.getOutputQueue("nn", maxSize=4, blocking=False)

    while True:
        image = qColor.get().getCvFrame()
        out = qNN.get()
        anomaly_map = np.array(out.getLayerFp16("anomaly_map")).reshape((1,1,256,256))
        pred_score = np.array(out.getLayerFp16("pred_score"))
        pred_mask = (anomaly_map >= 0.75).squeeze().astype(np.uint8)

        anomaly_map = anomaly_map.squeeze()
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)
        visualization = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

        image = cv2.resize(image, (832,832))

        visualization = mark_boundaries(
            visualization, pred_mask, color=(0, 0, 1), mode="thick"
        )
        visualization = (cv2.resize(visualization, (image.shape[1],image.shape[0]))*255).astype(np.uint8)

        # visualization = cv2.putText(visualization, f"{round(100*pred_score[0])}%", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)
        show = cv2.addWeighted(image, 0.6, visualization, 0.4, 0)
        show = cv2.hconcat([image, show])

        cv2.imshow('Anomaly Map', show)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
