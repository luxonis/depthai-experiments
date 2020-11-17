### Gen2 Camera Demo

StereoDepth configuration options:
```
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
```

If one or more of the additional depth modes (lrcheck, extended, subpixel) are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional. But like on Gen1, either depth or disparity has valid data. TODO enable both.


Select one pipeline to run:
```
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline, streams = create_stereo_depth_pipeline()
```

### Example depth results with subpixel and lrcheck enabled

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)
![image](https://user-images.githubusercontent.com/32992551/99454680-fea75b00-28e3-11eb-80bc-2004016d75e2.png)
![image](https://user-images.githubusercontent.com/32992551/99454698-0404a580-28e4-11eb-9cda-462708ef160d.png)
![image](https://user-images.githubusercontent.com/32992551/99454589-dfa8c900-28e3-11eb-8464-e719302d9f04.png)
