[中文文档](README.zh-CN.md)

## Camera Demo
This example shows how to use the DepthAI/megaAI/OAK cameras in the Gen2 Pipeline Builder over USB.  

Unfiltered subpixel disparity depth results from a [BW1092 board](https://shop.luxonis.com/collections/all/products/bw1092) over USB2:

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)

## Usage

### Navigate to directory

```bash
  cd ./api
```

### Install Dependencies
`python3 install_requirements.py`

> Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   

### Running Example As-Is
`python3 main.py` - Runs without point cloud visualization
`python3 main.py -pcl` - Enables point cloud visualization


This will run subpixel disparity by default.

### Real-Time Depth from DepthAI Stereo Pair

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

#### Example depth results with subpixel and lrcheck enabled

![image](https://user-images.githubusercontent.com/32992551/99454680-fea75b00-28e3-11eb-80bc-2004016d75e2.png)
![image](https://user-images.githubusercontent.com/32992551/99454698-0404a580-28e4-11eb-9cda-462708ef160d.png)
![image](https://user-images.githubusercontent.com/32992551/99454589-dfa8c900-28e3-11eb-8464-e719302d9f04.png)

### Depth from Rectified Host Images

Set `source_camera = False`

Using input images from: https://vision.middlebury.edu/stereo/data/scenes2014/

![image](https://user-images.githubusercontent.com/60824841/99694663-589b5280-2a95-11eb-94fe-3f9cc2afc158.png)

![image](https://user-images.githubusercontent.com/60824841/99694401-0eb26c80-2a95-11eb-8728-403665024750.png)

For the bad looking areas, these are caused by the objects being too close to the camera for the given baseline, exceeding the 96 pixels max distance for disparity matching (StereoDepth engine constraint):
![image](https://user-images.githubusercontent.com/60824841/99696549-7cf82e80-2a97-11eb-9dbd-3e3645be210f.png)

These areas will be improved with `extended = True`, however Extended Disparity and Subpixel cannot operate both at the same time.

