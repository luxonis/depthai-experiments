import logging
import sys
import depthai as dai
from pathlib import Path

from utils.constants import Config
from utils.arguments import initialize_argparser
from utils.audio_encoder import AudioEncoder
from utils.whisper_decoder import WhisperDecoder
from utils.annotation_node import AnnotationNode

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

if platform != "RVC4":
    raise ValueError("This experiment is only supported for RVC4 platform.")

encoder_model_description = dai.NNModelDescription(
    model="whisper-tiny-en:encoder:1.0.0",
    platform=platform,
)
encoder_archive_path = dai.getModelFromZoo(
    encoder_model_description,
    useCached=True,
)

decoder_model_description = dai.NNModelDescription(
    model="whisper-tiny-en:decoder:1.0.0",
    platform=platform,
)
decoder_archive_path = dai.getModelFromZoo(
    decoder_model_description,
    useCached=True,
)


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    # Encoder setup

    camera = pipeline.create(dai.node.Camera).build()
    camera_out = camera.requestOutput((1080, 720), dai.ImgFrame.Type.NV12, fps=30)

    audio_encoder = pipeline.create(AudioEncoder, args.audio_file)

    encoder_nn = pipeline.create(dai.node.NeuralNetwork)
    encoder_nn.setNNArchive(dai.NNArchive(archivePath=encoder_archive_path))
    audio_encoder.output.link(encoder_nn.input)

    # Decoder setup
    dec_host_data = pipeline.create(WhisperDecoder, Config.MEAN_DECODE_LEN)
    encoder_nn.out.link(dec_host_data.input_enc)

    decoder_nn = pipeline.create(dai.node.NeuralNetwork)
    decoder_nn.setNNArchive(dai.NNArchive(archivePath=decoder_archive_path))

    dec_host_data.out.link(decoder_nn.input)
    decoder_nn.out.link(dec_host_data.input_dec)

    # Script node for LED
    led_changer = pipeline.create(dai.node.Script)
    dec_host_data.out_color.link(led_changer.inputs["color_in"])
    led_changer.setScriptPath(Path(__file__).parent / "utils/led_changer_script.py")

    text_annotation = pipeline.create(AnnotationNode)
    camera_out.link(text_annotation.frame_intput)
    dec_host_data.annotation_out.link(text_annotation.color_text_input)

    visualizer.addTopic("Camera", text_annotation.frame_out)

    # Add visualizer topic
    visualizer.addTopic("Decoded Audio Message", text_annotation.annotaion_out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
        else:
            audio_encoder.handle_key_press(key)
