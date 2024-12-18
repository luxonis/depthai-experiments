import time
import depthai as dai
from depthai_nodes import ParserGenerator, XFeatMonoParser
from utils.custom_visualizer import MonoVersionVisualizer

def mono_mode(device: dai.Device, nn_archive: dai.NNArchive, visualizer: dai.RemoteConnection, fps_limit: int = 30):
    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")
        
        cam = pipeline.create(dai.node.Camera).build()
        
        platform = device.getPlatform().name
        print(f"Platform: {platform}")
        img_frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i

        network = pipeline.create(dai.node.NeuralNetwork).build(
            cam.requestOutput(
                nn_archive.getInputSize(), type=img_frame_type, fps=fps_limit
            ),
            nn_archive,
        )
        
        parser: XFeatMonoParser = pipeline.create(ParserGenerator).build(nn_archive)[0]
        network.out.link(parser.input)

        custom_visualizer = pipeline.create(MonoVersionVisualizer)
        parser.out.link(custom_visualizer.tracked_features)
        network.passthrough.link(custom_visualizer.target_frame)

        visualizer.addTopic("Matches", custom_visualizer.out, "images")

        print("Pipeline created.")
        print("\nPress 's' to set the reference frame.\n")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            time.sleep(1 / 30)
            key_pressed = visualizer.waitKey(1)
            if key_pressed == ord("s"):
                parser.setTrigger()
                custom_visualizer.setReferenceFrame()
            elif key_pressed == ord("q"):
                pipeline.stop()
                break
