[英文文档](README.md)

# Gen2 三角剖分 3D可视化器

因为通常需要在立体声上进行特定于应用程序的主机端过滤
神经推理结果，并且因为这些计算是轻量级的
（即可以在ESP32上完成），则将三角剖分本身留给主机。

该3D可视化器用于面部标志演示，并使用OpenGL和OpenCV。
此时将其视为草稿/参考。

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 演示

[![Spatial AI](https://user-images.githubusercontent.com/18037362/116149182-bc2b4b00-a6d9-11eb-91a5-ad5359ca85ad.gif)](https://www.youtube.com/watch?v=YalHMcsZODs&feature=youtu.be "3D Facial Landmark visualization")

## 安装依赖

```
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

请注意，此实验使用 `Script` 当前处于Alpha模式的节点，因此您必须下载最新的 `gen2-scripting` GitHub分支 (你通过下载得到它 `requirements.txt`)

## 用法

运行示例

```
python3 main.py
```

您应该会看到5个窗口:
- `mono_left` 它将显示来自左单声道相机+面部边界框和面部标志的相机输出
- `mono_right` 它将显示来自右单声道相机+面部边界框和面部标志的相机输出
- `crop_left` 这将显示进入第二个NN的48x48左裁剪图像+从第二个NN输出的面部标志
- `crop_right` 这将显示进入第二个NN的48x48右裁剪图像+从第二个NN输出的面部标志
- `pygame window` 这将显示三角剖分结果
