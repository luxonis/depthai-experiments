[英文文档](README.md)

# Tensorflow 图像分类示例

该示例演示了如何运行使用 [TensorFlow Image Classification tutorial](https://colab.research.google.com/drive/1oNxfvx5jOfcmk1Nx0qavjLN8KtWcLRn6?usp=sharing)创建的神经网络 （即使将OpenVINO转换为.blob，我们的社区成员之一也将其整合到了一个Colab Notebook中）


## 演示

![Pedestrian Re-Identification](https://user-images.githubusercontent.com/5244214/109003919-522a0180-76a8-11eb-948c-a74432c22be1.gif)

## 先决条件

1. 购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
```
python -m pip install -r requirements.txt
```

## 用法

```
用法: main.py [-h] [-nd] [-cam] [-vid VIDEO]

可选参数:
-h, --help      显示此帮助消息并退出
-nd, --no-debug 禁止调试输出
-cam, --camera  使用DepthAI 4K RGB相机进行推理(与-vid冲突)
-vid VIDEO, --video VIDEO
                用于推理的视频文件的路径(与-cam冲突)
```

### 使用设备运行程序

```
python main.py -cam
```

### 使用视频运行程序
``` 
python main.py -vid <path>
```

按"q"退出程序。