[英文文档](README.md)

# Class Saver (JPEG)

本示例演示如何运行MobilenetSSD并收集按检测标签分组的检测对象的图像。运行此脚本后，DepthAI将启动MobilenetSSD，并且每当检测到对象时，它将添加一个数据集条目。

数据集存储在`data`目录下，而主数据集文件位于下`data.dataset.csv`。对于每个检测到的对象，将创建一个新的数据集条目，每个条目都包含具有精确用途的文件:

- `timestamp` 用于 `int(time.time() * 10000)` 存储捕获的时间戳。 **请注意** 如果在单个图像上检测到多个对象，则可以重复使用此值。
- `label` 是检测到的物体的人类可读标签。
- `left`, `top`, `right`, `top` 是对象边界框坐标。
- `raw_frame` 表示检测到发生时在DepthAI上捕获的原始RGB帧的路径。

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018096-47163c80-67a0-11eb-88f6-c67fb3c2f421.jpg)
  
- `overlay_frame` 表示具有检测覆盖图(边界框和标签)的RGB帧的路径

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018179-63b27480-67a0-11eb-8423-4fd311a6d860.jpg)

- `cropped_frame` 表示仅包含所检测对象的ROI的裁剪RGB帧的路径

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018256-7dec5280-67a0-11eb-964e-2cc08b6b75fd.jpg)

示例条目`dataset.csv`如下所示

```
timestamp,label,left,top,right,bottom,raw_frame,overlay_frame,cropped_frame
16125187249289,bottle,0,126,79,300,data/raw/16125187249289.jpg,data/bottle/16125187249289_overlay.jpg,data/bottle/16125187249289_cropped.jpg
16125187249289,person,71,37,300,297,data/raw/16125187249289.jpg,data/person/16125187249289_overlay.jpg,data/person/16125187249289_cropped.jpg
16125187249653,bottle,0,126,79,300,data/raw/16125187249653.jpg,data/bottle/16125187249653_overlay.jpg,data/bottle/16125187249653_cropped.jpg
16125187249653,person,71,36,300,297,data/raw/16125187249653.jpg,data/person/16125187249653_overlay.jpg,data/person/16125187249653_cropped.jpg
16125187249992,bottle,0,126,80,300,data/raw/16125187249992.jpg,data/bottle/16125187249992_overlay.jpg,data/bottle/16125187249992_cropped.jpg
16125187249992,person,71,37,300,297,data/raw/16125187249992.jpg,data/person/16125187249992_overlay.jpg,data/person/16125187249992_cropped.jpg
16125187250374,person,37,38,300,299,data/raw/16125187250374.jpg,data/person/16125187250374_overlay.jpg,data/person/16125187250374_cropped.jpg
16125187250769,bottle,0,126,79,300,data/raw/16125187250769.jpg,data/bottle/16125187250769_overlay.jpg,data/bottle/16125187250769_cropped.jpg
16125187250769,person,71,36,299,297,data/raw/16125187250769.jpg,data/person/16125187250769_overlay.jpg,data/person/16125187250769_cropped.jpg
16125187251120,bottle,0,126,80,300,data/raw/16125187251120.jpg,data/bottle/16125187251120_overlay.jpg,data/bottle/16125187251120_cropped.jpg
16125187251120,person,77,37,300,298,data/raw/16125187251120.jpg,data/person/16125187251120_overlay.jpg,data/person/16125187251120_cropped.jpg
16125187251492,bottle,0,126,79,300,data/raw/16125187251492.jpg,data/bottle/16125187251492_overlay.jpg,data/bottle/16125187251492_cropped.jpg
16125187251492,person,74,38,300,297,data/raw/16125187251492.jpg,data/person/16125187251492_overlay.jpg,data/person/16125187251492_cropped.jpg
```

## 演示

[![Gen2 Class Saver (JPEG)](https://user-images.githubusercontent.com/5244214/106964520-83b34b00-6742-11eb-8729-eff0a7584a46.gif)](https://youtu.be/gKawPaUcTi4 "Class Saver (JPEG) on DepthAI")

## 先决条件

1. 购买DepthAI模型 (请参考 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
python3 main.py
```

数据集将存储在`data`目录中
