import re
import logging
import depthai as dai
import numpy as np
from scipy import special as scipy_special
from tqdm import tqdm
from whisper.decoding import get_tokenizer

from depthai_nodes.utils import AnnotationHelper

from utils.constants import Config

logger = logging.getLogger()


class TextMessage(dai.Buffer):
    def __init__(self, text: str = "", color: tuple = ()) -> None:
        super().__init__()
        self.text = text
        self.color = color


class WhisperDecoder(dai.node.ThreadedHostNode):
    """Processes decoded outputs into readable text."""

    def __init__(self, sample_len):
        super().__init__()
        self.tokenizer = get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )
        self.input_enc = self.createInput()
        self.input_dec = self.createInput()

        self.out = self.createOutput()
        self.annotation_out = self.createOutput()
        self.out_color = self.createOutput()

        self.sample_len = sample_len
        self.index = np.zeros([1, 1], dtype=np.int32)
        self.encoder_outputs = []
        self.decoded_tokens = [Config.TOKENS["TOKEN_SOT"]]

        self._annotation_helper = AnnotationHelper()

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

    def parser_text(self, text: str) -> tuple[str, str]:
        """Parses text into tuples of (str, color) and returns the new color of LED (None if not detected or multiple detected)."""
        words = text.split()
        words = [re.sub(r"[^\w]", "", word.lower()) for word in words]

        word_colors = [
            (
                word,
                Config.LED_COLORS.get(word, (255, 255, 255)),
            )
            for word in words
        ]
        # keep only words valid colors
        word_colors = [
            (word, color)
            for word, color in word_colors
            if word in Config.LED_COLORS.keys()
        ]

        if len(word_colors) == 0:
            print("No color keywords found in the text.")
            return "", []
        word_color = word_colors[0]

        color = word_color[1][::-1]
        word = word_color[0]

        return word, color

    def run(self) -> None:
        """Run the decoder and process encoder outputs."""
        text = None
        while self.isRunning():
            for i in tqdm(range(self.sample_len), total=self.sample_len):
                # Get the encoder outputs on the first iteration
                if not self.encoder_outputs:
                    print("dont have encoder outputs")
                    encoder_out = self.input_enc.get()
                    ts = encoder_out.getTimestamp()
                    seq_num = encoder_out.getSequenceNum()
                    print("names", encoder_out.getAllLayerNames())
                    self.encoder_outputs = [
                        encoder_out.getTensor(name)
                        for name in encoder_out.getAllLayerNames()
                    ]
                    assert (
                        len(self.encoder_outputs) == 2
                    ), f"Expected 2 encoder outputs, got {len(self.encoder_outputs)}"

                    decoder_k_cache = np.zeros(
                        (4, 6, 64, Config.MEAN_DECODE_LEN), dtype=np.float16
                    )
                    decoder_v_cache = np.zeros(
                        (4, 6, Config.MEAN_DECODE_LEN, 64), dtype=np.float16
                    )
                    tokens = self.get_tokens(i)
                    print(f"[{seq_num}] encoder_outputs: tokens", tokens)

                else:
                    print("have encoder outputs")
                    decoder_out = self.input_dec.get()
                    decoder_outputs = {
                        name: decoder_out.getTensor(name)
                        for name in decoder_out.getAllLayerNames()
                    }
                    assert (
                        len(decoder_outputs) == 3
                    ), f"Expected 3 decoder outputs, got {len(decoder_outputs)}"

                    decoder_k_cache = decoder_outputs["k_cache"]
                    decoder_v_cache = decoder_outputs["v_cache"]
                    logits = decoder_outputs["logits"]
                    tokens = self.get_tokens(i, logits)

                if tokens is None:
                    text = self.tokenizer.decode(self.decoded_tokens[1:])
                    logging.info(f"Decoding completed at iteration {i}. Text: {text}")
                    word_colors, color = self.parser_text(text)

                    if len(color) == 3:
                        msg = dai.NNData()
                        msg.addTensor(
                            "color",
                            np.array(color, dtype=np.uint8),
                            dataType=dai.TensorInfo.DataType.INT,
                        )
                        msg.setSequenceNum(seq_num)
                        msg.setTimestamp(ts)

                        self.out_color.send(msg)

                    text_msg = TextMessage(word_colors, color)
                    self.annotation_out.send(text_msg)
                    break

                # print("encoder outputs to send to decoder:", self.encoder_outputs)
                encoder_k_cache, encoder_v_cache = self.encoder_outputs

                self.index[0, 0] = i
                decoder_input = dai.NNData()
                decoder_input.addTensor(
                    "k_cache_cross",
                    encoder_k_cache,
                    dataType=dai.TensorInfo.DataType.FP16,
                )
                decoder_input.addTensor(
                    "v_cache_cross",
                    encoder_v_cache,
                    dataType=dai.TensorInfo.DataType.FP16,
                )
                decoder_input.addTensor(
                    "k_cache_self",
                    decoder_k_cache,
                    dataType=dai.TensorInfo.DataType.FP16,
                )
                decoder_input.addTensor(
                    "v_cache_self",
                    decoder_v_cache,
                    dataType=dai.TensorInfo.DataType.FP16,
                )
                decoder_input.addTensor(
                    "x", tokens, dataType=dai.TensorInfo.DataType.INT
                )
                decoder_input.addTensor(
                    "index", self.index, dataType=dai.TensorInfo.DataType.INT
                )
                self.out.send(decoder_input)
