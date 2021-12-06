from depthai_sdk import Previews, FPSHandler
from depthai_sdk.managers import PipelineManager, PreviewManager, BlobManager, NNetManager
import depthai as dai
import cv2
import argparse
from pathlib import Path

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model path for inference",
                    default='models/yolo_v3_tiny_openvino_2021.3_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='yolo-tiny.json', type=str)
args = parser.parse_args()
CONFIG_PATH = args.config

# create blob, NN, and preview managers
if Path(args.model).exists():
    bm = BlobManager(blobPath=args.model)
else:
    raise ValueError("Only local models are implemented in the SDK for now. "
                     "Instead, please try using main.py for models from the DepthAI ZOO.")
    #bm = BlobManager(zooName=args.model)

nm = NNetManager(nnFamily="YOLO", inputSize=4)
nm.readConfig(CONFIG_PATH)  # this will also parse the correct input size

pm = PipelineManager()
pm.createColorCam(previewSize=nm.inputSize, xout=True)

# create preview manager
fpsHandler = FPSHandler()
pv = PreviewManager(display=[Previews.color.name], scale={"color":0.33}, fpsHandler=fpsHandler)

# create NN with managers
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion(), zoo_type="depthai"))
pm.addNn(nn)

# initialize pipeline
with dai.Device(pm.pipeline) as device:
    # create outputs
    pv.createQueues(device)
    nm.createQueues(device)

    nnData = []

    while True:
        # count FPS
        fpsHandler.tick("color")

        # parse outputs
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            nnData = nm.decode(inNn)

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break

