from typing import List, Tuple
import numpy as np
import cv2
import depthai as dai
from on_device_pipeline import DeviceEncoder
import av

# Total compressed frame size and the uncompressed frames
def get_compressed_frames_host_jpeg(input_frames: List[np.ndarray], quality: int = 100) -> Tuple[int, List[np.ndarray]]:
    return_frames = []
    total_size_output_frames = 0
    for input_frame in input_frames:
        if input_frame.dtype != np.uint8:
            raise ValueError("Input frame must be uint8")
        if input_frame.ndim != 3:
            raise ValueError("Input frame must be 3-dimensional")
        if input_frame.shape[2] != 3:
            raise ValueError("Input frame must have 3 channels")
        # Compress frame
        _, output_frame = cv2.imencode(".jpg", input_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        # Add to total size
        total_size_output_frames += output_frame.size
        # Decode frame
        output_frame = cv2.imdecode(output_frame, cv2.IMREAD_COLOR)
        return_frames.append(output_frame)
    return total_size_output_frames, return_frames

def get_compressed_frames_device_jpeg(input_frames: List[np.ndarray], quality: int = 100, lossless = True, bitrate_kbps = 1000) -> Tuple[int, List[np.ndarray]]:
    # TODO add device implementation
    return_frames = []
    total_size_output_frames = 0
    deviceEncoder = DeviceEncoder(input_frames[0].shape[1], input_frames[0].shape[0], dai.VideoEncoderProperties.Profile.MJPEG, lossless, bitrate_kbps, quality)
    codec = av.CodecContext.create("mjpeg", "r")
    for input_frame in input_frames:
        if input_frame.dtype != np.uint8:
            raise ValueError("Input frame must be uint8")
        if input_frame.ndim != 3:
            raise ValueError("Input frame must be 3-dimensional")
        if input_frame.shape[2] != 3:
            raise ValueError("Input frame must have 3 channels")
        deviceEncoder.send_frame(input_frame)
        # Compress frame
        output_frame = deviceEncoder.get_frame()
        # print(output_frame.shape)
        # Add to total size
        total_size_output_frames += output_frame.size
        codec.parse(None) # Flush
        packets = codec.parse(output_frame)
        if not packets:
            packets = codec.parse(output_frame)
            if not packets:
                raise RuntimeError("No packets!")
        for packet in packets:
            output_frame = codec.decode(packet)[0].to_ndarray(format="bgr24")
        # Decode frame
        # output_frame = cv2.imdecode(output_frame, cv2.IMREAD_COLOR)
        return_frames.append(output_frame)
    return total_size_output_frames, return_frames

def get_compressed_frames_device_h264(input_frames: List[np.ndarray], quality: int = 100, lossless = True, bitrate_kbps = 1000) -> Tuple[int, List[np.ndarray]]:
    return_frames = []
    total_size_output_frames = 0

    # Initialize the device encoder for H.264
    deviceEncoder = DeviceEncoder(input_frames[0].shape[1], input_frames[0].shape[0], dai.VideoEncoderProperties.Profile.H264_MAIN, lossless, bitrate_kbps)

    # Initialize codec for H.264
    codec = av.CodecContext.create("h264", "r")

    for input_frame in input_frames:
        if input_frame.dtype != np.uint8:
            raise ValueError("Input frame must be uint8")
        if input_frame.ndim != 3:
            raise ValueError("Input frame must be 3-dimensional")
        if input_frame.shape[2] != 3:
            raise ValueError("Input frame must have 3 channels")

        deviceEncoder.send_frame(input_frame)

        # Compress frame
        output_frame = deviceEncoder.get_frame()

        # Add to total size
        total_size_output_frames += output_frame.size

        # Parse and decode the frame
        codec.parse(None)  # Flush
        packets = codec.parse(output_frame)
        if not packets:
            packets = codec.parse(output_frame)
            if not packets:
                raise RuntimeError("No packets!")

        for packet in packets:
            output_frame = codec.decode(packet)[0].to_ndarray(format="bgr24")

        return_frames.append(output_frame)

    return total_size_output_frames, return_frames

def get_compressed_frames_device_h265(input_frames: List[np.ndarray], quality: int = 100, lossless = True, bitrate_kbps = 1000) -> Tuple[int, List[np.ndarray]]:
    return_frames = []
    total_size_output_frames = 0

    # Initialize the device encoder for H.264
    deviceEncoder = DeviceEncoder(input_frames[0].shape[1], input_frames[0].shape[0], dai.VideoEncoderProperties.Profile.H265_MAIN, lossless, bitrate_kbps)

    # Initialize codec for H.264
    codec = av.CodecContext.create("hevc", "r")

    for i, input_frame in enumerate(input_frames):
        if input_frame.dtype != np.uint8:
            raise ValueError("Input frame must be uint8")
        if input_frame.ndim != 3:
            raise ValueError("Input frame must be 3-dimensional")
        if input_frame.shape[2] != 3:
            raise ValueError("Input frame must have 3 channels")

        deviceEncoder.send_frame(input_frame)

        # Compress frame
        output_frame = deviceEncoder.get_frame()

        # Add to total size
        total_size_output_frames += output_frame.size

        # Parse and decode the frame
        # codec.parse(None)  # Flush
        packets = codec.parse(output_frame)
        if not packets:
            continue
            packets = codec.parse(output_frame)
            if not packets:
                raise RuntimeError("No packets!")
        # print(i)
        output_frame = None
        # print("Length packets: ", len(packets))
        for packet in packets:
            output_frames = codec.decode(packet)
            if output_frames:
                output_frame = output_frames[0].to_ndarray(format="bgr24")
            # else:
            #     output_frame = np.zeros((input_frame.shape[0], input_frame.shape[1], 3), dtype=np.uint8)
        if output_frame is None:
            continue
        return_frames.append(output_frame)

    return total_size_output_frames, return_frames