import time
import depthai as dai
from depthai_nodes import ParserGenerator, XFeatStereoParser
from utils.custom_visualizer import StereoVersionVisualizer

def stereo_mode(device: dai.Device, nn_archive: dai.NNArchive, visualizer: dai.RemoteConnection, fps_limit: int = 30):
    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        platform = device.getPlatform().name
        print(f"Platform: {platform}")
        img_frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i

        available_cameras = [
            camera.name for camera in device.getConnectedCameraFeatures()
        ]

        if "left" not in available_cameras or "right" not in available_cameras:
            raise RuntimeError(
                f"Stereo cameras are not available! Available cameras: {available_cameras}"
            )

        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_network = pipeline.create(dai.node.NeuralNetwork).build(
            left_cam.requestOutput(
                nn_archive.getInputSize(), type=img_frame_type, fps=fps_limit
            ),
            nn_archive,
        )
        left_network.setNumInferenceThreads(2)

        right_network = pipeline.create(dai.node.NeuralNetwork).build(
            right_cam.requestOutput(
                nn_archive.getInputSize(), type=img_frame_type, fps=fps_limit
            ),
            nn_archive,
        )
        right_network.setNumInferenceThreads(2)

        parser: XFeatStereoParser = pipeline.create(ParserGenerator).build(nn_archive)[0]
        parser.setMaxKeypoints(512)
        left_network.out.link(parser.reference_input)
        right_network.out.link(parser.target_input)

        custom_visualizer = pipeline.create(StereoVersionVisualizer)
        parser.out.link(custom_visualizer.tracked_features)
        left_network.passthrough.link(custom_visualizer.left_frame_input)
        right_network.passthrough.link(custom_visualizer.right_frame_input)

        visualizer.addTopic("Left Camera", left_network.passthrough, "images")
        visualizer.addTopic("Right Camera", right_network.passthrough, "images")
        visualizer.addTopic("Matches", custom_visualizer.out, "images")

        print("Pipeline created.")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            time.sleep(1 / 30)
            key_pressed = visualizer.waitKey(1)
            if key_pressed == ord("q"):
                pipeline.stop()
                break