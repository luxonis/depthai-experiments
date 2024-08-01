import depthai as dai
import blobconverter

from host_seven_segment_recognition import SevenSegmentRecognition

nn_shape = (750, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(600, 600)
    cam.setFps(20)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    cam.video.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="7_segment_recognition_256x750", zoo_type="depthai", shaves=6))
    manip.out.link(nn.input)

    segment_recognition = pipeline.create(SevenSegmentRecognition).build(
        preview=cam.video,
        nn=nn.out
    )
    segment_recognition.inputs["preview"].setBlocking(False)
    segment_recognition.inputs["preview"].setMaxSize(4)
    segment_recognition.inputs["nn"].setBlocking(False)
    segment_recognition.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
