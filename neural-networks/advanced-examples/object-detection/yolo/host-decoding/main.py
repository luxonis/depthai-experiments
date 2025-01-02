import argparse

import depthai as dai
from host_decoding import HostDecoding

parser = argparse.ArgumentParser()

parser.add_argument(
    "-conf",
    "--confidence_thresh",
    help="set the confidence threshold",
    default=0.5,
    type=float,
)
parser.add_argument(
    "-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.45, type=float
)


args = parser.parse_args()

conf_thresh = args.confidence_thresh
iou_thresh = args.iou_thresh


model_description = dai.NNModelDescription(
    modelSlug="yolov6-nano", platform="RVC2", modelVersionSlug="r2-coco-512x288"
)
archive_path = dai.getModelFromZoo(model_description, useCached=True)
nn_archive = dai.NNArchive(archive_path)

CAM_SIZE = (1280, 720)
NN_SIZE = (512, 288)

visualizer = dai.RemoteConnection()

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(CAM_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)
    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(*NN_SIZE)
    color_out.link(manip.inputImage)
    nn = pipeline.create(dai.node.NeuralNetwork).build(manip.out, nn_archive)

    host_decoding = pipeline.create(HostDecoding).build(nn=nn.out)
    host_decoding.set_nn_size(NN_SIZE)
    host_decoding.set_conf_thresh(conf_thresh)
    host_decoding.set_iou_thresh(iou_thresh)

    visualizer.addTopic("Camera", color_out)
    visualizer.addTopic("Detections", host_decoding.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
