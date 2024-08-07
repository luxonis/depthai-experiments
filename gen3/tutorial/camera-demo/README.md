## Gen3 Camera Demo
This example shows how to use the DepthAI/megaAI/OAK cameras in the Gen3 Pipeline Builder over USB.  

### Install Dependencies:
`python3 install_requirements.py`

### Running Example As-Is:
`python3 main.py` - run

This will run subpixel disparity by default.

### Real-Time Depth from DepthAI Stereo Pair

StereoDepth configuration options:
```
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
```

Select one pipeline to run:
```
    # test_mono_cam_pipeline(pipeline)
    # test_rgb_cam_pipeline(pipeline)
    test_stereo_depth_pipeline(pipeline)
```
