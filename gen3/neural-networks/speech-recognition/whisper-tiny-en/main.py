import argparse
import logging
import os
import re
import sys
import time
from enum import IntEnum

import cv2
import depthai as dai
import numpy as np
from scipy import special as scipy_special
from tqdm import tqdm
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper.decoding import get_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class Tokens(IntEnum):
    TOKEN_SOT = 50257  # Start of transcript
    TOKEN_EOT = 50256  # End of transcript
    TOKEN_BLANK = 220  # Blank token " "
    TOKEN_NO_TIMESTAMP = 50362
    TOKEN_TIMESTAMP_BEGIN = 50363
    TOKEN_NO_SPEECH = 50361


class Config:
    """Configuration constants."""

    NON_SPEECH_TOKENS = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91,
        92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303,
        1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600,
        4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907,
        13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724,
        22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282,
        49146, 50257, 50357, 50358, 50359, 50360, 50361,
    ]
    TOKENS = Tokens
    SAMPLE_BEGIN = 1  # first token is TOKEN_SOT
    NO_SPEECH_THR = 0.6  # Above this prob we deem there's no speech in the audio
    PRECISION = 0.02  # in seconds
    MAX_INITIAL_TIMESTAMP = 1.0  # in seconds
    MAX_INITIAL_TIMESTAMP_INDEX = int(MAX_INITIAL_TIMESTAMP / PRECISION)
    MEAN_DECODE_LEN = 224  # The official default max decoded length is 448.
    LED_COLORS = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
        "pink": (203, 192, 255),
        "purple": (128, 0, 128),
        "brown": (19, 69, 139),
    }


class AudioProcessor:
    """Processes audio files to create spectrograms for the encoder."""

    @staticmethod
    def process_audio(file_path: str) -> np.ndarray:
        audio = load_audio(file_path)
        audio = pad_or_trim(audio)
        mel_spectrogram = log_mel_spectrogram(audio)
        mel_spectrogram = mel_spectrogram.unsqueeze(0).numpy().astype(np.float16)
        assert mel_spectrogram.shape == (1, 80, 3000), (
            f"Expected shape (1, 80, 3000), got {mel_spectrogram.shape}"
        )
        logging.info(
            f"Processed audio file {file_path}. Spectrogram shape: {mel_spectrogram.shape}"
        )
        return mel_spectrogram


class EncoderHostData(dai.node.ThreadedHostNode):
    """Encodes audio into spectrogram format."""

    def __init__(self, audio_file="/app/assets/audio_files/command_LED_yellow.mp3"):
        super().__init__()
        self.out = self.createOutput()
        self.audio_file = audio_file

    def run(self) -> None:
        mel_spectrogram = AudioProcessor.process_audio(self.audio_file)
        nn_data = dai.NNData()
        nn_data.addTensor(
            "audio", mel_spectrogram, dataType=dai.TensorInfo.DataType.FP16
        )
        self.out.send(nn_data)


class DecoderHostData(dai.node.ThreadedHostNode):
    """Processes decoded outputs into readable text."""

    def __init__(self, sample_len):
        super().__init__()
        self.tokenizer = get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )
        self.input_enc = self.createInput()
        self.input_dec = self.createInput()
        self.out = self.createOutput()
        self.out_text = self.createOutput()
        self.sample_len = sample_len
        self.index = np.zeros([1, 1], dtype=np.int32)
        self.encoder_outputs = []
        self.decoded_tokens = [Config.TOKENS["TOKEN_SOT"]]

    def apply_timestamp_rules(self, logits: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply timestamp-related post-processing rules to logits."""

        # Require producing timestamp
        logits[Config.TOKENS.TOKEN_NO_TIMESTAMP] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT
        seq = self.decoded_tokens[Config.SAMPLE_BEGIN :]
        last_was_timestamp = (
            len(seq) >= 1 and seq[-1] >= Config.TOKENS.TOKEN_TIMESTAMP_BEGIN
        )
        penultimate_was_timestamp = (
            len(seq) < 2 or seq[-2] >= Config.TOKENS.TOKEN_TIMESTAMP_BEGIN
        )
        if last_was_timestamp:
            if penultimate_was_timestamp:  # has to be non-timestamp
                logits[Config.TOKENS.TOKEN_TIMESTAMP_BEGIN :] = -np.inf
            else:  # cannot be normal text tokens
                logits[: Config.TOKENS.TOKEN_EOT] = -np.inf

        timestamps = [
            t for t in self.decoded_tokens if t >= Config.TOKENS.TOKEN_TIMESTAMP_BEGIN
        ]
        if len(timestamps) > 0:
            # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
            # also force each segment to have a nonzero length, to   prevent infinite looping
            if last_was_timestamp and not penultimate_was_timestamp:
                timestamp_last = timestamps[-1]
            else:
                timestamp_last = timestamps[-1] + 1
            logits[Config.TOKENS.TOKEN_TIMESTAMP_BEGIN : timestamp_last] = -np.inf

        if len(self.decoded_tokens) == Config.SAMPLE_BEGIN:
            # suppress generating non-timestamp tokens at the beginning
            logits[: Config.TOKENS.TOKEN_TIMESTAMP_BEGIN] = -np.inf

            # apply the `max_initial_timestamp` option
            last_allowed = (
                Config.TOKENS.TOKEN_TIMESTAMP_BEGIN + Config.MAX_INITIAL_TIMESTAMP_INDEX
            )
            logits[(last_allowed + 1) :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = scipy_special.log_softmax(logits)
        timestamp_logprob = scipy_special.logsumexp(
            logprobs[Config.TOKENS.TOKEN_TIMESTAMP_BEGIN :]
        )
        # logprobs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
        # timestamp_logprob = np.log(np.sum(np.exp(logprobs[Config.TOKENS.TOKEN_TIMESTAMP_BEGIN:])))
        max_text_token_logprob = logprobs[: Config.TOKENS.TOKEN_TIMESTAMP_BEGIN].max()
        if timestamp_logprob > max_text_token_logprob:
            # Mask out all but timestamp tokens
            logits[: Config.TOKENS.TOKEN_TIMESTAMP_BEGIN] = -np.inf

        return logits, logprobs

    def get_tokens(self, index, logits=None):
        """Get the next tokens based on current logits."""

        if index == 0:
            return np.array([[Config.TOKENS.TOKEN_SOT]], dtype=np.int32)
        logits = logits[0, -1]  # Process the last token's logits

        # Filters
        # SuppressBlank
        if index == 1:
            logits[[Config.TOKENS.TOKEN_EOT, Config.TOKENS.TOKEN_BLANK]] = -np.inf
        # SuppressTokens
        logits[Config.NON_SPEECH_TOKENS] = -np.inf

        logits, logprobs = self.apply_timestamp_rules(logits)

        if index == 1:
            # detect no_speech
            no_speech_prob = np.exp(logprobs[Config.TOKENS.TOKEN_NO_SPEECH])
            if no_speech_prob > Config.NO_SPEECH_THR:
                return None

        # temperature = 0
        next_token = np.argmax(logits)
        if next_token == Config.TOKENS.TOKEN_EOT:
            return None

        x = np.array([[next_token]], dtype=np.int32)
        self.decoded_tokens.append(int(next_token))
        return x

    def visualize_text(self, text):
        """Visualize the transcribed text with LED color handling."""
        frame_width, frame_height = 500, 80
        font_scale, speed = 1, 4
        font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2

        # Create a black frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Split the text into words and check for color keywords
        words = text.split()
        word_colors = [
            (
                word,
                Config.LED_COLORS.get(
                    re.sub(r"[^\w]", "", word.lower()), (255, 255, 255)
                ),
            )
            for word in words
        ]

        unique_colors = [color for _, color in word_colors if color != (255, 255, 255)]
        if len(unique_colors) == 1 or all(
            color == unique_colors[0] for color in unique_colors
        ):
            color = unique_colors[0][::-1]  # Convert BGR to RGB
            os_command = f"echo 1,{color[0]},{color[1]},{color[2]} > /dev/status_led"
            logging.info(f"Setting LED to {color}. Command: {os_command}")
            os.system(os_command)
        else:
            logging.info(
                "No unique LED color or multiple colors found; no LED changes made."
            )

        x, y = frame_width, (frame_height // 2) + 10
        while True:
            time.sleep(0.03)
            frame[:] = 0
            current_x = x

            for word, color in word_colors:
                text_size = cv2.getTextSize(word + " ", font, font_scale, thickness)[0]
                word_width = text_size[0]

                cv2.putText(
                    frame, word, (current_x, y), font, font_scale, color, thickness
                )
                current_x += word_width

                if current_x > frame_width:
                    break

            x -= speed
            if current_x < 0:
                x = frame_width

            img_frame = dai.ImgFrame()
            img_frame.setFrame(frame)
            img_frame.setWidth(frame_width)
            img_frame.setHeight(frame_height)
            img_frame.setType(dai.ImgFrame.Type.BGR888i)
            self.out_text.send(img_frame)

    def run(self):
        """Run the decoder and process encoder outputs."""
        inference_start_time = time.time()
        text = None

        for i in tqdm(range(self.sample_len), total=self.sample_len):
            # Get the encoder outputs on the first iteration
            if not self.encoder_outputs:
                encoder_out = self.input_enc.get()
                self.encoder_outputs = [
                    encoder_out.getTensor(name)
                    for name in encoder_out.getAllLayerNames()
                ]
                assert len(self.encoder_outputs) == 2, (
                    f"Expected 2 encoder outputs, got {len(self.encoder_outputs)}"
                )

                decoder_k_cache = np.zeros(
                    (4, 6, 64, Config.MEAN_DECODE_LEN), dtype=np.float16
                )
                decoder_v_cache = np.zeros(
                    (4, 6, Config.MEAN_DECODE_LEN, 64), dtype=np.float16
                )
                tokens = self.get_tokens(i)

            else:
                decoder_out = self.input_dec.get()
                decoder_outputs = {
                    name: decoder_out.getTensor(name)
                    for name in decoder_out.getAllLayerNames()
                }
                assert len(decoder_outputs) == 3, (
                    f"Expected 3 decoder outputs, got {len(decoder_outputs)}"
                )

                decoder_k_cache = decoder_outputs["k_cache"]
                decoder_v_cache = decoder_outputs["v_cache"]
                logits = decoder_outputs["logits"]
                tokens = self.get_tokens(i, logits)

            if tokens is None:
                text = self.tokenizer.decode(self.decoded_tokens[1:])
                logging.info(f"Decoding completed at iteration {i}. Text: {text}")
                break

            encoder_k_cache, encoder_v_cache = self.encoder_outputs

            self.index[0, 0] = i
            decoder_input = dai.NNData()
            decoder_input.addTensor(
                "k_cache_cross", encoder_k_cache, dataType=dai.TensorInfo.DataType.FP16
            )
            decoder_input.addTensor(
                "v_cache_cross", encoder_v_cache, dataType=dai.TensorInfo.DataType.FP16
            )
            decoder_input.addTensor(
                "k_cache_self", decoder_k_cache, dataType=dai.TensorInfo.DataType.FP16
            )
            decoder_input.addTensor(
                "v_cache_self", decoder_v_cache, dataType=dai.TensorInfo.DataType.FP16
            )
            decoder_input.addTensor("x", tokens, dataType=dai.TensorInfo.DataType.INT)
            decoder_input.addTensor(
                "index", self.index, dataType=dai.TensorInfo.DataType.INT
            )
            self.out.send(decoder_input)

        logging.info(f"Decoding completed in {time.time() - inference_start_time:.3f}s")
        if text:
            self.visualize_text(text)
        else:
            logging.warning("No text to visualize.")


class PipelineManager:
    """Sets up and manages the entire pipeline."""

    def __init__(self, device_ip, audio_file):
        self.device_ip = device_ip
        self.audio_file = audio_file

    def get_model_from_hub(self, model_slug, platform):
        logging.info(
            f"Downloading model {model_slug} from the hub for platform {platform}"
        )
        model_description = dai.NNModelDescription(
            model=model_slug,
            platform=platform.name,
        )
        model_archive_path = dai.getModelFromZoo(
            model_description,
            useCached=True,
        )

        return model_archive_path

    def run(self):
        pipeline_setup_time = time.time()

        if self.device_ip == "":
            device = dai.Device()
        else:
            device = dai.Device(dai.DeviceInfo(self.device_ip))
        platform = device.getPlatform()
        logging.info(f"Detected platform: {platform}")

        # Get the model archives from the hub
        encoder_archive_path = self.get_model_from_hub(
            "whisper-tiny-en:encoder:1.0.0", platform
        )
        decoder_archive_path = self.get_model_from_hub(
            "whisper-tiny-en:decoder:1.0.0", platform
        )

        # Setup visualizer
        visualizer = dai.RemoteConnection()

        with dai.Pipeline(device) as pipeline:
            # Encoder setup
            enc_host_data = pipeline.create(EncoderHostData, self.audio_file)
            encoder_nn = pipeline.create(dai.node.NeuralNetwork)
            encoder_nn.setNNArchive(dai.NNArchive(archivePath=encoder_archive_path))
            encoder_nn.setBackend("snpe")
            encoder_nn.setBackendProperties({
                "runtime": "dsp",  # "cpu" if using unquantized model, "dsp" if using quantized model
                "performance_profile": "default",
            })

            # Decoder setup
            dec_host_data = pipeline.create(DecoderHostData, Config.MEAN_DECODE_LEN)
            decoder_nn = pipeline.create(dai.node.NeuralNetwork)
            decoder_nn.setNNArchive(dai.NNArchive(archivePath=decoder_archive_path))
            decoder_nn.setBackend("snpe")
            decoder_nn.setBackendProperties({
                "runtime": "dsp",  # "cpu" if using unquantized model, "dsp" if using quantized model
                "performance_profile": "default",
            })

            # Linking components
            enc_host_data.out.link(encoder_nn.input)
            encoder_nn.out.link(dec_host_data.input_enc)
            dec_host_data.out.link(decoder_nn.input)
            decoder_nn.out.link(dec_host_data.input_dec)

            # Add visualizer topic
            visualizer.addTopic("Decoded Audio Message", dec_host_data.out_text)

            logging.info(
                f"Pipeline setup completed in {time.time() - pipeline_setup_time:.3f}s"
            )

            pipeline.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Tiny EN Demo")
    parser.add_argument(
        "--device_ip",
        type=str,
        required=True,
        help="The IP address of the target device",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="The path to the audio file to process",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device_ip = args.device_ip
    audio_file_path = args.audio_file

    pipeline_manager = PipelineManager(device_ip, audio_file_path)
    pipeline_manager.run()
