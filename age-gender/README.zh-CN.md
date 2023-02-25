[英文文档](README.md)

# 年龄与性别识别

本示例演示了如何使用 Pipeline Builder在DepthAI上运行两阶段推理。首先，在图像上检测到面部，然后将裁剪的面部框发送到年龄性别识别网络，该网络产生估计结果


## 演示

[![Age & Gender recognition](https://user-images.githubusercontent.com/5244214/106005496-954a8200-60b4-11eb-923e-b84df9de9fff.gif)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## 先决条件

1. 购买DepthAI模型 (请参考 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
参数: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            显示此帮助消息并退出
  -nd, --no-debug       不显示调试输出
  -cam, --camera        使用DepthAI 4K RGB相机进行推理(与-vid冲突)
  -vid VIDEO, --video VIDEO
                        用于推理的视频文件的路径(与-cam冲突)
```

要与视频文件一起使用，请使用以下参数运行脚本

```
python3 main.py -vid ./input.mp4
```

要与DepthAI 4K RGB相机一起使用，请改用

```
python3 main.py -cam
``` 
