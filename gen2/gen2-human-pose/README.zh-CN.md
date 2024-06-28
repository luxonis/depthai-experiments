[英文文档](README.md)

# Gen2 姿势估算示例

本示例演示了如何使用Gen2 Pipeline Builder运行 [人体姿态估计网络](https://docs.openvinotoolkit.org/latest/omz_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html)


## 演示

### 相机
[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/107493701-35f97100-6b8e-11eb-8b13-02a7a8dbec21.gif)](https://www.youtube.com/watch?v=Py3-dHQymko "Human pose estimation on DepthAI")

### 视频

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/110801736-d3bf8900-827d-11eb-934b-9755978f80d9.gif)](https://www.youtube.com/watch?v=1dp2wJ_OqxI "Human pose estimation on DepthAI")


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
   -h, --help            显示此帮助消息并退出
   -nd, --no-debug       禁止调试输出
   -cam, --camera        使用DepthAI 4K RGB相机进行推理(与-vid冲突)
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
