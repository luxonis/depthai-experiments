import logging
import sys
import depthai as dai
from pathlib import Path

from utils.constants import Config
from utils.arguments import initialize_argparser
from utils.helper_nodes import EncoderHostData, DecoderHostData

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
    enc_host_data = pipeline.create(EncoderHostData, args.audio_file)
    encoder_nn = pipeline.create(dai.node.NeuralNetwork)
    encoder_nn.setNNArchive(dai.NNArchive(archivePath=encoder_archive_path))
    encoder_nn.setBackend("snpe")
    encoder_nn.setBackendProperties(
        {
            "runtime": "dsp",  # "cpu" if using unquantized model, "dsp" if using quantized model
            "performance_profile": "default",
        }
    )
    enc_host_data.out.link(encoder_nn.input)

    # Decoder setup
    dec_host_data = pipeline.create(DecoderHostData, Config.MEAN_DECODE_LEN)
    encoder_nn.out.link(dec_host_data.input_enc)

    decoder_nn = pipeline.create(dai.node.NeuralNetwork)
    decoder_nn.setNNArchive(dai.NNArchive(archivePath=decoder_archive_path))
    decoder_nn.setBackend("snpe")
    decoder_nn.setBackendProperties(
        {
            "runtime": "dsp",  # "cpu" if using unquantized model, "dsp" if using quantized model
            "performance_profile": "default",
        }
    )
    dec_host_data.out.link(decoder_nn.input)
    decoder_nn.out.link(dec_host_data.input_dec)

    # Script node for LED
    led_changer = pipeline.create(dai.node.Script)
    dec_host_data.out_color.link(led_changer.inputs["color_in"])
    led_changer.setScriptPath(Path(__file__).parent / "utils/led_changer_script.py")
    led_changer.setLogLevel(dai.LogLevel.WARN)

    # Add visualizer topic
    visualizer.addTopic("Decoded Audio Message", dec_host_data.out_text)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
