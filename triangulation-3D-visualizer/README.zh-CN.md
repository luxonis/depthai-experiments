[英文文档](README.md)

# 三角剖分3D可视化器

因为通常需要在立体神经推理结果上进行特定于应用程序的主机端过滤，并且由于这些计算是轻量级的（即可以在ESP32上完成），因此将三角剖分本身留给了主机。
如果有兴趣直接在DepthAI上执行此操作，请告诉我们！

该3D可视化器用于面部标志演示，并使用OpenGL和OpenCV。
此时将其视为草稿/参考。

## 演示

[![Spatial AI](https://user-images.githubusercontent.com/5244214/90748450-7c9e9a80-e2d2-11ea-9e9e-da65b5d9e6f0.gif)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")

请注意，当我们记录此内容时，我们意外地反转了结果-嘴巴的关键点在最上方，而眼睛在最下方。


## 安装依赖

```
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

## 用法

运行应用程序

```
python3 main.py
```

您应该看到出现5个窗口:
- `previewout-left` 这将显示左单声道相机的相机输出
- `previewout-right` 这将显示右单声道相机的相机输出
- `left` 这将显示基于左单帧的神经网络结果
- `right` 这将显示基于正确的单帧的神经网络结果
- `pygame window` 这将显示三角剖分结果
