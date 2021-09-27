#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np

from depthai_sdk import PipelineManager, NNetManager, PreviewManager, Previews, FPSHandler, toTensorResult

nn_shape = 896, 512

def decode(packet):
    data = np.squeeze(toTensorResult(packet)["L0317_ReWeight_SoftMax"])
    class_colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    indices = np.argmax(data, axis=0)
    output_colors = np.take(class_colors, indices, axis=0)
    return output_colors


def draw(data, frame):
    if len(data) == 0:
        return
    cv2.addWeighted(frame, 1, cv2.resize(data, frame.shape[:2][::-1]), 0.2, 0, frame)


# Start defining a pipeline
pm = PipelineManager()
pm.createColorCam(previewSize=nn_shape)

nm = NNetManager(inputSize=nn_shape)
pm.setNnManager(nm)
pm.addNn(
    nm.createNN(pm.pipeline, pm.nodes, blobconverter.from_zoo(name='road-segmentation-adas-0001', shaves=6)),
    sync=True
)
fps = FPSHandler()
pv = PreviewManager(display=[Previews.color.name], fpsHandler=fps)

# Pipeline is defined, now we can connect to the device
with dai.Device(pm.pipeline) as device:
    nm.createQueues(device)
    pv.createQueues(device)

    while True:
        fps.tick('color')
        pv.prepareFrames(blocking=True)
        frame = pv.get(Previews.color.name)

        road_decoded = decode(nm.outputQueue.get())
        draw(road_decoded, frame)
        fps.drawFps(frame, 'color')
        cv2.imshow('color', frame)

        if cv2.waitKey(1) == ord('q'):
            break
