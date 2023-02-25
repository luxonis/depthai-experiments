[英文文档](README.md)

人脸检测
================

该示例演示了Gen2 Pipeline Builder运行的人[face detection network](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_retail_0004.html)  ,[head posture estimation network](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html) 和 [face recognition network](https://docs.openvinotoolkit.org/latest/omz_models_model_face_recognition_mobilefacenet_arcface.html)


[![Face recognition](https://user-images.githubusercontent.com/18037362/134054837-eed40899-7c1d-4160-aaf0-1d7c405bb7f4.gif)](https://www.youtube.com/watch?v=HNAeBwNCRek "Face recognition")

## 这个怎么运作:

### 1. 运行人脸检测模型

> 运行 [face-detection-retail-0004](models/face-detection-retail-0004_openvino_2020_1_4shave.blob) 模型以检测图像中的面部并截取面部图像。
> 
> ![detection_face](assets/detection_face.png)

### 2. 运行头姿势估计模型

> 运行 [head-pose-estimation-adas-0001](models/head-pose-estimation-adas-0001.blob)模型以检测头部倾斜角度并调整头部姿势。
> 
>![face_corr](assets/face_corr.png)

### 3. 跑步人脸识别模型

> 运行 [face-recognition-mobilefacenet-arcface.blob](models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob) 模型以识别人脸。
>
> ![face_reg](assets/face_reg.png)

--------------------

## 先决条件

1.购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```bash
   python3 -m pip install -r requirements.txt
   ```


## 用法

```bash
用法: main.py [-h] [-nd] [-cam] [-vid VIDEO]

可选参数:
  -h, --help            显示此帮助消息并退出
  -nd, --no-debug       阻止调试输出
  -cam, --camera        使用DepthAI 4K RGB相机进行推理(与-vid冲突)
  -vid VIDEO, --video VIDEO
                        用于推理的视频文件的路径(与-cam冲突)
  -db, --databases      保存数据(仅在运行识别网络时使用)
  -n NAME, --name NAME  NAME数据名称(与-db一起使用)[可选]

```

与视频文件一起使用并建立人脸数据库
```bash
python main.py -db -n <name> -vid <path>
```

要与DepthAI 4K RGB相机一起使用，请改用
```bash
python main.py -db -n <name> -cam
```

要与视频文件一起使用，请使用以下参数运行脚本

```bash
python main.py -vid <path>
```

要与DepthAI 4K RGB相机一起使用，请改用
```bash
python main.py -cam
```

> 按“ q”退出程序。
