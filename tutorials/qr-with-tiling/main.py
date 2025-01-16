import depthai as dai
from pathlib import Path
from host_qr_scanner import QRScanner

from depthai_nodes.ml.helpers import Tiling, TilesPatcher

model_description = dai.NNModelDescription(modelSlug="qrdet", platform="RVC2")
archivePath = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath)

with dai.Pipeline() as pipeline:
    # IMG_SHAPE = (1080, 1080)
    # IMG_SHAPE = (1920, 1080)
    # IMG_SHAPE = (1280, 720)
    IMG_SHAPE = (3840, 2160)

    # cam = pipeline.create(dai.node.ColorCamera)
    # cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam.setPreviewSize(IMG_SHAPE)
    # cam.setInterleaved(False)
    # cam.initialControl.setManualFocus(145)
    # cam.setFps(20)

    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setLoop(False)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    replay.setReplayVideoFile(Path("videos/home_test_2.mp4"))
    replay.setSize(IMG_SHAPE)
    replay.setFps(10)
    cam_out = replay.out

    grid_size = (9, 6)
    # grid_size = (1,1) # no tiling :D
    grid_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    tile_manager = pipeline.create(Tiling).build(
        img_output=cam_out,
        img_shape=IMG_SHAPE,
        overlap=0.2,
        grid_size=grid_size,
        grid_matrix=grid_matrix,
        global_detection=False,
        nn_shape=(512, 288),
    )

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        nnArchive=nn_archive, input=tile_manager.out, confidenceThreshold=0.3
    )
    nn.input.setMaxSize(len(tile_manager.tile_positions))
    nn.input.setBlocking(False)

    patcher = pipeline.create(TilesPatcher).build(
        tile_manager=tile_manager, nn=nn.out, conf_thresh=0.3, iou_thresh=0.2
    )

    scanner = pipeline.create(QRScanner).build(
        preview=cam_out, nn=patcher.out, tile_positions=tile_manager.tile_positions
    )
    scanner.inputs["detections"].setBlocking(False)
    scanner.inputs["detections"].setMaxSize(2)
    scanner.inputs["preview"].setBlocking(False)
    scanner.inputs["preview"].setMaxSize(2)

    print("Pipeline created.")
    pipeline.run()
