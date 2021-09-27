[英文文档](README.md)

# Gen2 NN帧同步

这个例子展示了如何在被推断的帧上呈现神经网络的推断结果。使用了两种帧同步方式:
- 使用DepthAI设备端队列进行设备同步(以这种方式显示人脸检测结果)
- 使用内置`queue`模块进行主机同步(以这种方式显示地标检测结果)

## 演示

![image](https://user-images.githubusercontent.com/5244214/104956823-36f31480-59cd-11eb-9568-64c0f0003dd0.gif)


## 设置

```
python3 -m pip -U pip
python3 -m pip install -r requirements.txt
```

## 运行

```
python3 main.py
```