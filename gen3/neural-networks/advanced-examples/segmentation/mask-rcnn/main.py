import blobconverter
import depthai as dai
import argparse

from host_mask_rcnn import MaskRCNN

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.5, type=float
                    , help="Threshold for filtering out detections with lower probability")
parser.add_argument('-rt', '--region_threshold', default=0.5, type=float
                    , help="Threshold for filtering out mask points with low probability")
args = parser.parse_args()

nn_shape = (300, 300)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(nn_shape)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo("mask_rcnn_resnet50_coco_300x300", shaves=6, zoo_type="depthai"))
    cam.preview.link(nn.input)

    maskrcnn = pipeline.create(MaskRCNN).build(
        preview=cam.preview,
        nn=nn.out,
        threshold=args.threshold,
        region_threshold=args.region_threshold
    )
    maskrcnn.inputs["preview"].setBlocking(False)
    maskrcnn.inputs["preview"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
