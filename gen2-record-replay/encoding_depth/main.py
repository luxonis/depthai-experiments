import argparse
from pathlib import Path
import numpy as np
import cv2
from depth_loader import DepthLoader
from color_mapping import HueColorMap, MonoColorMap
from comparison import print_all_metrics, get_union_mask
from compression import get_compressed_frames_host_jpeg, get_compressed_frames_device_jpeg, get_compressed_frames_device_h264, get_compressed_frames_device_h265

SHIFT_DISP = 100
argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--dataset_path", default="./testData", help="Path to the dataset")
argparser.add_argument("-nf", "--num_frames", default=1, type=int, help="Number of frames to load")
argparser.add_argument("-mask", "--mask", action="store_true", help="Mask to only check valid depth pixels")
argparser.add_argument("-q", "--quality", default=100, type=int, help="Quality of the JPEG compression")
argparser.add_argument("-l", "--lossless", action="store_true", help="Use lossless compression (only for device compression)")
argparser.add_argument("-c", "--codec", default="jpeg", choices=["jpeg", "h264", "h265"], help="Codec to use for compression")
argparser.add_argument("-br", "--bitrate", default=8000, type=int, help="Bitrate to use for compression (only for h264 and h265) in kbps")
argparser.add_argument("--timeout_preview", default=0, type=int, help="Timeout for the preview window")
args = argparser.parse_args()
WAIT_TIME = args.timeout_preview

def main():
    depth_loader = DepthLoader(args.dataset_path)
    if args.num_frames > depth_loader.json_info["num_frames"]:
        raise ValueError(f"Number of frames to load ({args.num_frames}) is greater than the number of frames in the dataset ({depth_loader.json_info['num_frames']})")
    first_depth = depth_loader.get_next_frame()
    max_disparity = first_depth.max_disparity
    color_map = HueColorMap(-100, max_disparity + 100)
    # color_map = MonoColorMap(0, max_disparity)

    # Prepare input frames
    input_disparity_frames = []
    input_frames = []
    total_input_size = 0
    for i in range(args.num_frames):
        frame = depth_loader.get_next_frame()
        disparity = frame.get_disparity()
        disparity = disparity
        input_disparity_frames.append(disparity)
        colorized_disparity_frame = color_map.to_color(disparity)
        input_frames.append(colorized_disparity_frame)
        total_input_size += input_disparity_frames[-1].size

    # Compress the frames
    if args.codec == "jpeg":
        # compressed_size, output_frames = get_compressed_frames_device_jpeg(input_frames, quality=args.quality, lossless=args.lossless)
        compressed_size, output_frames = get_compressed_frames_device_jpeg(input_frames, quality=args.quality, lossless=args.lossless)
    elif args.codec == "h264":
        compressed_size, output_frames = get_compressed_frames_device_h264(input_frames, quality=args.quality, lossless=args.lossless, bitrate_kbps=args.bitrate)
    elif args.codec == "h265":
        compressed_size, output_frames = get_compressed_frames_device_h265(input_frames, quality=args.quality, lossless=args.lossless, bitrate_kbps=args.bitrate)
    output_disparity_frames = []
    for output_frame in output_frames:
        output_disparity_frames.append(color_map.to_mono(output_frame))
    print(f"Total input size: {total_input_size}")
    print(f"Total compressed size: {compressed_size}")
    print(f"Compression ratio: {total_input_size / compressed_size}")

    # If more output disparity frames than input disparity frames, remove the extra input frames
    print("Length of input disparity frames: ", len(input_disparity_frames))
    print("Length of output disparity frames: ", len(output_disparity_frames))
    if len(input_disparity_frames) > len(output_disparity_frames):
        input_disparity_frames = input_disparity_frames[:len(output_disparity_frames) + 1]

    matrix_json_average = None
    for in_frame, out_frame in zip(input_disparity_frames, output_disparity_frames):
        mask = get_union_mask(in_frame, out_frame, threshold=0)
        if not args.mask:
            mask = None
        cv2.imshow("Original disparity", in_frame.astype(np.uint8))
        cv2.imshow("Compressed disparity", out_frame.astype(np.uint8))
        print("Max disparity", max_disparity)
        matrix_json = print_all_metrics(in_frame, out_frame, max_disparity, mask)
        if matrix_json_average is None:
            matrix_json_average = matrix_json
        else:
            for key in matrix_json:
                matrix_json_average[key] += matrix_json[key]
        cv2.waitKey(WAIT_TIME)
    for key in matrix_json_average:
        matrix_json_average[key] /= len(input_disparity_frames)
    print("Average metrics:")
    print(matrix_json_average)
    # Create the csv file
    csv_file_path = Path(args.dataset_path) / "compression_metrics.csv"
    if not csv_file_path.exists():
        with open(csv_file_path, "w") as f:
            f.write("codec,quality,lossless,bitrate,psnr,mse,average_difference,compression_ratio,max_difference,original_fillrate,compressed_fillrate,relative_fillrate\n")
    # Append the metrics to the csv file
    with open(csv_file_path, "a") as f:
        f.write(f"{args.codec},{args.quality},{args.lossless},{args.bitrate},{matrix_json_average['psnr']},{matrix_json_average['mse']},{matrix_json_average['average_difference']},{total_input_size / compressed_size},{matrix_json_average['max_difference']},{matrix_json_average['original_fillrate']},{matrix_json_average['compressed_fillrate']},{matrix_json_average['relative_fillrate']}\n")

if __name__ == "__main__":
    main()