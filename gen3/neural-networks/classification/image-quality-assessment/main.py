import depthai as dai
import argparse
import blobconverter
from host_image_quality_assessment import ImageQualityAssessment

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn_path', type=str, help="select model blob path for inference, defaults to image_quality_assessment_256x256_001 from model zoo", default=None)
args = parser.parse_args()

nn_path = args.nn_path if args.nn_path else blobconverter.from_zoo(name="image_quality_assessment_256x256", zoo_type="depthai", shaves=6)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(256, 256)
    cam.setInterleaved(False)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(nn_path)
    cam.preview.link(nn.input)

    quality_assessment = pipeline.create(ImageQualityAssessment).build(
        preview=cam.preview,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
