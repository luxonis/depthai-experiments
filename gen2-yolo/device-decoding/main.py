from depthai_sdk import Previews, FPSHandler
from depthai_sdk.managers import PipelineManager, PreviewManager, BlobManager, NNetManager
import depthai as dai
import cv2
import argparse
from pathlib import Path

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
args = parser.parse_args()
CONFIG_PATH = args.config

# create blob, NN, and preview managers
if Path(args.model).exists():
    # initialize blob manager with path to the blob
    bm = BlobManager(blobPath=args.model)
else:
    # initialize blob manager with the name of the model otherwise
    bm = BlobManager(zooName=args.model)

nm = NNetManager(nnFamily="YOLO", inputSize=4)
nm.readConfig(CONFIG_PATH)  # this will also parse the correct input size

pm = PipelineManager()
pm.createColorCam(previewSize=nm.inputSize, xout=True)

# create preview manager
fpsHandler = FPSHandler()
pv = PreviewManager(display=[Previews.color.name], fpsHandler=fpsHandler)

# create NN with managers
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion(), zooType="depthai"))
pm.addNn(nn)

# initialize pipeline
with dai.Device(pm.pipeline) as device:
    # create outputs
    pv.createQueues(device)
    nm.createQueues(device)

    nnData = []

    while True:

        # parse outputs
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            nnData = nm.decode(inNn)
            # count FPS
            fpsHandler.tick("color")

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break

