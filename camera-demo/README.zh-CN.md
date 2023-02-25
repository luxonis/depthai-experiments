[英文文档](README.md)

## 相机演示
此示例显示了如何通过USB在Gen2 Pipeline Builder中使用DepthAI / megaAI / OAK摄像机。

未过滤的亚像素视差深度是由USB2上的 [BW1092 board](https://shop.luxonis.com/collections/all/products/bw1092)产生的:

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)

### 安装依赖项:

`python3 install_requirements.py`

注意: `python3 install_requirements.py` 还会尝试从requirements-optional.txt中安装可选的库。例如：它包含open3d lib，这对于点云可视化是必需的。但是，该库的二进制文件不适用于树莓派和jetson之类的某些主机。

### 按原样运行示例:

`python3 main.py` - 在没有点云可视化的情况下运行

`python3 main.py -pcl` - 启用点云可视化默认情况下，这将运行亚像素视差。

### DepthAI双目的实时深度探测

StereoDepth 配置选项:
```
lrcheck  = True   # 更好地处理咬合
extended = False  # 接近最小深度，视差范围加倍
subpixel = True   # 更好的精度，可实现更长的距离，分数差异为32级
```

如果启用了一个或多个其他深度模式(lrcheck，扩展，子像素)，则:
 - 深度输出为FP16。TODO启用U16。
 - 在设备上禁用了中值过滤。启用TODO。
 - 对于亚像素，深度或视差具有有效数据。

否则，深度输出为U16(mm)，中位数有效。但是像Gen1一样，深度或视差都具有有效数据。TODO都启用。

选择一个要运行的管道:

```
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline, streams = create_stereo_depth_pipeline()
```

#### 启用亚像素和lrcheck的示例深度结果

![image](https://user-images.githubusercontent.com/32992551/99454680-fea75b00-28e3-11eb-80bc-2004016d75e2.png)
![image](https://user-images.githubusercontent.com/32992551/99454698-0404a580-28e4-11eb-9cda-462708ef160d.png)
![image](https://user-images.githubusercontent.com/32992551/99454589-dfa8c900-28e3-11eb-8464-e719302d9f04.png)

### 校正主机图像的深度

设置 `source_camera = False`

使用来自以下位置的输入图像: https://vision.middlebury.edu/stereo/data/scenes2014/

![image](https://user-images.githubusercontent.com/60824841/99694663-589b5280-2a95-11eb-94fe-3f9cc2afc158.png)

![image](https://user-images.githubusercontent.com/60824841/99694401-0eb26c80-2a95-11eb-8728-403665024750.png)

对于外观不好的区域，这是由于对于给定的基线，对象太靠近相机，导致视差匹配超过了96个像素的最大距离(StereoDepth引擎约束): (StereoDepth engine constraint):
![image](https://user-images.githubusercontent.com/60824841/99696549-7cf82e80-2a97-11eb-9dbd-3e3645be210f.png)

这些区域将通过进行改进`extended = True`，但是“扩展视差”和“子像素”不能同时运行。

