from depthai_sdk import Previews, FPSHandler
from depthai_sdk.managers import PipelineManager, PreviewManager, BlobManager, NNetManager
import depthai as dai
import cv2
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model path for inference",
                    default='models/yolo_v3_tiny_openvino_2021.3_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='yolo-tiny.json', type=str)
parser.add_argument("--height", help="Input shape height", default=320, type=int)
parser.add_argument("--width", help="Input shape width", default=512, type=int)
args = parser.parse_args()

CONFIG_PATH = args.config
H, W = args.height, args.width

# create pipeline manager and camera
pm = PipelineManager()
pm.createColorCam(previewSize=(W, H), xout=True)

# create yolo node
bm = BlobManager(blobPath=args.model)
nm = NNetManager(inputSize=(W, H), nnFamily="YOLO")
nm.readConfig(CONFIG_PATH)
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion()))
pm.addNn(nn)

# initialize pipeline
with dai.Device(pm.pipeline) as device:

    fpsHandler = FPSHandler()
    pv = PreviewManager(display=[Previews.color.name], scale={"color":0.33}, fpsHandler=fpsHandler)

    pv.createQueues(device)
    nm.createQueues(device)

    nnData = []

    while True:
        # parse outputs
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            # count FPS
            fpsHandler.tick("color")

            nnData = nm.decode(inNn)

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break

