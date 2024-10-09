import depthai as dai
from depthai_nodes.ml.parsers import SCRFDParser
from host_node.depth_merger import DepthMerger
from host_node.draw_detections import DrawDetections
from host_node.frame_stacker import FrameStacker
from host_node.host_bird_eye_view import BirdsEyeView
from host_node.host_display import Display
from host_node.measure_object_distance import MeasureObjectDistance
from host_node.normalize_bbox import NormalizeBbox
from host_node.parser_bridge import ParserBridge
from host_social_distancing import SocialDistancing

FPS = 10

device = dai.Device()

modelDescription = dai.NNModelDescription(
    modelSlug="scrfd-person-detection",
    platform=device.getPlatform().name,
    modelVersionSlug="2-5g-640x640",
)
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)
nnArchive = dai.NNArchive(archivePath)


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    rgb = cam.requestOutput(
        size=(640, 640),
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS,
    )

    left = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_B
    )
    right = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_C
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput((640, 640), fps=FPS),
        right=right.requestOutput((640, 640), fps=FPS),
    )
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    stereo.setOutputSize(640, 640)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nnArchive)
    nn.input.setBlocking(False)
    nn_parser = pipeline.create(SCRFDParser)
    nn_parser.setFeatStrideFPN((8, 16, 32, 64, 128))
    nn_parser.setNumAnchors(1)
    nn.out.link(nn_parser.input)
    parser_bridge = pipeline.create(ParserBridge).build(nn_parser.out)

    depth_merger = pipeline.create(DepthMerger).build(
        output_2d=nn_parser.out,
        output_depth=stereo.depth,
        calib_data=device.readCalibration2(),
    )

    rgb.link(nn.input)

    bird_eye_view = pipeline.create(BirdsEyeView).build(depth_merger.output)
    normalize_bboxes = pipeline.create(NormalizeBbox).build(
        frame=rgb, nn=parser_bridge.output
    )
    measure_obj_dist = pipeline.create(MeasureObjectDistance).build(depth_merger.output)
    draw_bboxes = pipeline.create(DrawDetections).build(
        frame=rgb, nn=normalize_bboxes.output, label_map=["person"]
    )
    social_distancing_new = pipeline.create(SocialDistancing).build(
        frame=draw_bboxes.output, distances=measure_obj_dist.output
    )
    frame_stacker = pipeline.create(FrameStacker).build(
        frame_1=social_distancing_new.output, frame_2=bird_eye_view.output
    )

    display = pipeline.create(Display).build(frames=frame_stacker.output)
    display.setName("Social distancing")

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline exited.")
