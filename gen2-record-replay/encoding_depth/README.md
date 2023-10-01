# On host depth/disparity encoding and decoding

On device encoding for now only supports 8-bit depth encoding. 16-bit depth encoding is not supported.
This direcotry contains the host side encoding and decoding for 16-bit depth/disparity.
I this written in a relatively modular way, so different color maps should be easy to test out.


## Requirements
Before running any of the examples in this directory, requirements have to be installed with
```
pip install -r requirements.txt
```

## Usage
First thing that has te be done is recording of the depth frames in the expected format.
This can be done with the `record.py` script. This script will record the depth frames.
For more info see the help of the script.
```
usage: recorder.py [-h] [-nf NUM_FRAMES] [-e] [-sub] [-lr] [-path PATH] [--subpixel_bits SUBPIXEL_BITS]

options:
  -h, --help            show this help message and exit
  -nf NUM_FRAMES, --num_frames NUM_FRAMES
                        Number of frames to record
  -e, --extended        Enable extended disparity
  -sub, --subpixel      Enable subpixel disparity
  -lr, --lr_check       Enable left-right check
  -path PATH, --path PATH
                        Path where to store the frames
  --subpixel_bits SUBPIXEL_BITS
                        Subpixel disparity bits
```

After recording the frames, the frames can be encoded/decoded and metrics calculated with main.py.
For more info see:
```
usage: main.py [-h] [-p DATASET_PATH] [-nf NUM_FRAMES] [-mask] [-q QUALITY] [-l] [-c {jpeg,h264,h265}] [-br BITRATE]

options:
  -h, --help            show this help message and exit
  -p DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset
  -nf NUM_FRAMES, --num_frames NUM_FRAMES
                        Number of frames to load
  -mask, --mask         Mask to only check valid depth pixels
  -q QUALITY, --quality QUALITY
                        Quality of the JPEG compression
  -l, --lossless        Use lossless compression (only for device compression)
  -c {jpeg,h264,h265}, --codec {jpeg,h264,h265}
                        Codec to use for compression
  -br BITRATE, --bitrate BITRATE
                        Bitrate to use for compression (only for h264 and h265) in kbp
```

Example usage:
```
python recorder.py --path ./test
python main.py --dataset_path ./test --num_frames 10 --codec jpeg --quality 90 --mask
```

