# Replay

This experiment shows how to save all camera streams (rgb and left/right mono), optionally encoded, on the host and then
replay the video with spatial object detection.

## Pre-requisites

```
python3 -m pip install -r requirements.txt
```

## Record usage
```
usage: record.py [-h] [-t THRESHOLD] [-p PATH] [-d] [-m TIME] [-af {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}] [-mf MANUALFOCUS] [-et EXPOSURE_TIME] [-ei EXPOSURE_ISO] [-e] [-disp]

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Maximum difference between packet timestamps to be considered as synced
  -p PATH, --path PATH  Path where to store the captured data
  -d, --dirty           Allow the destination path not to be empty
  -m TIME, --time TIME  Finish execution after X seconds
  -af {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}, --autofocus {OFF,AUTO,MACRO,CONTINUOUS_VIDEO,CONTINUOUS_PICTURE,EDOF}
                        Set AutoFocus mode of the RGB camera
  -mf MANUALFOCUS, --manualfocus MANUALFOCUS
                        Set manual focus of the RGB camera [0..255]
  -et EXPOSURE_TIME, --exposure-time EXPOSURE_TIME
                        Set manual exposure time of the RGB camera [1..33000]
  -ei EXPOSURE_ISO, --exposure-iso EXPOSURE_ISO
                        Set manual exposure ISO of the RGB camera [100..1600]
   -e, --encode         Encode mono frames into jpeg. If set, it will enable --mono as well
   -nd, --no-depth      Do not save depth map
   -mono, --mono        Save mono frames
```

By default, script will save encoded (jpeg) color frames and depth map.
## Replay usage
```
usage: replay.py [-p PATH]

optional arguments:
  -p PATH, --path PATH  Path where to store the captured data
```

Using the defaults, `record.py` will store dataset in the `data` directory
