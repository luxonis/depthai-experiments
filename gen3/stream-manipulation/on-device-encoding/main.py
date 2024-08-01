import argparse
import depthai as dai

from host_encoder import Encoder

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--codec", choices=["h264", "h265", "mjpeg"], default="h264", type=str
                    , help="Choose video encoding (h264 is default)")
args = parser.parse_args()

if args.codec == "h264":
    encoder_profile = dai.VideoEncoderProperties.Profile.H264_MAIN
elif args.codec == "h265":
    args.codec = "hevc"
    encoder_profile = dai.VideoEncoderProperties.Profile.H265_MAIN
elif args.codec == "mjpeg":
    encoder_profile = dai.VideoEncoderProperties.Profile.MJPEG

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setPreviewSize(1280, 720)
    cam.setFps(30)

    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(fps=30, profile=encoder_profile)
    cam.video.link(video_enc.input)

    encoder = pipeline.create(Encoder).build(
        preview=cam.preview,
        stream=video_enc.bitstream,
        codec=args.codec,
        output_shape=(3840, 2160)
    )
    encoder.inputs["preview"].setBlocking(False)
    encoder.inputs["preview"].setMaxSize(4)
    encoder.inputs["video_data"].setBlocking(True)
    encoder.inputs["video_data"].setMaxSize(30)

    print(f"Pipeline created. App starting streaming {encoder_profile.name} encoded frames into file video.mp4")
    pipeline.run()
