import depthai as dai
from pathlib import Path
from host_mega_depth import MegaDepth

nn_path = Path("models/megadepth_192x256_openvino_2021.4_6shave.blob").absolute().resolve()
nn_shape = (192, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(nn_shape[1], nn_shape[0])
    cam.setInterleaved(False)
    cam.setFps(10)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(nn_path)
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    cam.preview.link(nn.input)

    depth = pipeline.create(MegaDepth).build(
        preview=cam.preview,
        nn=nn.out,
        nn_shape=nn_shape
    )
    depth.inputs["preview"].setBlocking(False)
    depth.inputs["preview"].setMaxSize(4)
    depth.inputs["nn"].setBlocking(False)
    depth.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
