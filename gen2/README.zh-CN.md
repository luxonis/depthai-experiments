[depthai-experiments英文文档](README.md)

# depthai-experiments
我们使用DepthAI做的实验项目。

**示例** 在此库中包含了各个示例的代码，运行它们调用depthai。这几乎是一个depthai使用教程。

以下列表并不详尽(由于我们随机添加实验，可能会忘记更新此列表):

## [Gen2] 凝视估计 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-gaze-estimation))

该示例演示了如何使用 [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136) 在DepthAI上运行3个模型(3序列，2并行)推理。

[![Gaze Example Demo](https://user-images.githubusercontent.com/5244214/96713680-426c7a80-13a1-11eb-81e6-238e3decb7be.gif)](https://www.youtube.com/watch?v=OzgK5-APxBU)

制作此示例的Origina OpenVINO演示在[这里](https://github.com/LCTyrell/Gaze_pointer_controller)

## [Gen2] 亚像素和LR检查视差深度 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-camera-demo))

此示例显示了如何进行亚像素，LR检查或扩展视差，以及如何将这些测量值投影到点云中以进行可视化。这将使用 [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136)。

![image](https://user-images.githubusercontent.com/32992551/99454609-e59eaa00-28e3-11eb-8858-e82fd8e6eaac.png)
![image](https://user-images.githubusercontent.com/32992551/99454680-fea75b00-28e3-11eb-80bc-2004016d75e2.png)

## [Gen2] 年龄与性别检测 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender#gen2-age--gender-recognition))

这显示了一个简单的两阶段神经推理示例，先进行面部检测，然后根据面部进行年龄/性别估计。

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/5244214/106005496-954a8200-60b4-11eb-923e-b84df9de9fff.gif)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## [Gen2] 文本检测+光学字符识别(OCR)管道 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-ocr#gen2-text-detection--optical-character-recognition-ocr-pipeline))

该管道执行文本检测(EAST)，然后对检测到的文本进行光学字符识别。

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749667-f6315900-5f00-11eb-92bd-a297590adedc.png)](https://www.youtube.com/watch?v=YWIZYeixQjc "Gen2 OCR Pipeline")

## [Gen2] 行人识别 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification))

该示例演示了如何使用Gen2 Pipeline Builder在DepthAI上进行两阶段推理，以识别和重新识别具有唯一ID的行人。

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

原OpenVINO演示，在其上实现这个例子，详情在[这里](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).

## COVID-19口罩/无口罩检测器 ([点击查看详情](https://github.com/luxonis/depthai-experiments/blob/master/coronamask))

该项目向您展示了如何运行在[这里](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)训练过的COVID-19口罩/无口罩检测器。

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/90733159-74436100-e2cc-11ea-8fb6-d4be937d90e5.gif)](https://photos.app.goo.gl/mJZ8TdWoNatHzW4x7 "COVID-19 mask detection")

## 社会距离示例 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/social-distancing))

由于DepthAI可以提供对象在物理空间中的完整3D位置，因此使用DepthAI编写社交距离监视器需要几行代码。这就是这个项目，它是快速与社会隔离的监视器进行操作的地方。

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## 演示界面 ([点击查看详情](https://github.com/luxonis/depthai-python/tree/gen2_develop/examples))

该应用程序用于演示DepthAI平台的各种功能。包含带有说明，控制台输出和预览窗口的示例。

![DemoUI](./demo-ui/preview.png)

## MJPEG和JSON流式传输 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-mjpeg-streaming))

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

这为使DepthAI与[OpenDataCam](https://github.com/opendatacam/opendatacam)兼容奠定了基础。

## 立体声神经推理结果可视化工具 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/triangulation-3D-visualizer))

因此，由于通常在立体神经推理结果上进行特定于应用程序的主机端过滤，并且由于这些计算是轻量级的(即可以在ESP32上完成)，因此将三角剖分本身留给了主机。如果有兴趣直接在DepthAI上执行此操作，请告诉我们！

[![Spatial AI](https://user-images.githubusercontent.com/5244214/90748450-7c9e9a80-e2d2-11ea-9e9e-da65b5d9e6f0.gif)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")

## 人数统计 ([点击查看详情](https://github.com/luxonis/depthai-experiments/blob/master/people-counter))

这是megaAI和/或DepthAI的基本用法示例（尽管实际上并没有使用DepthAI的深度方面）：仅对场景中的人进行计数并记录该计数。

因此，您可以使用它来绘制一天的房间占用情况。人们可以修改这个例子来说明，其中在一个房间里那帮人，随着时间的推移，如果需要的话。但是目前，随着时间的推移，它只会产生一定数量的人员-因此，从摄像机的角度来看，总数是多少。

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## 人物追踪器 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker))

此应用程序统计视频流中有多少人向上/向下/向左/向右移动，从而使您可以接收有关多少人进入房间或​​经过走廊的信息。

在这个例子中使用的模型是 [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) 从OpenVIN模型动物园。

[![Watch the demo](https://user-images.githubusercontent.com/18037362/116413235-56e96e00-a82f-11eb-8007-bfcdb27d015c.gif)](https://www.youtube.com/watch?v=MHmzp--pqUA)


## 点云投影 ([点击查看详情](https://github.com/luxonis/depthai-experiments/blob/master/point-cloud-projection))

这是一个简单的应用程序，可从中创建rgbd图像`right`并进行`depth_raw`流传输并将其投影到点云中。还有一个交互式点云可视化器。（带有left和rgb的depth_raw将很快添加）

![point cloud visualization](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)


## RGB-D和PCL([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/pcl-projection-rgb))

这是一个简单的应用程序，可从中创建rgbd图像`rgb`并进行`depth`流处理，并将其投影到具有深度叠加和点云的rgb中。还有一个交互式点云可视化器。

![rgbd](https://media.giphy.com/media/SnW9p4r3feMQGOmayy/giphy.gif)
![rgbd-pcl](https://media.giphy.com/media/UeAlkPpeHaxItO0NJ6/giphy.gif)


## 主机端WLS过滤器 ([点击查看详情](https://github.com/luxonis/depthai-experiments/tree/master/gen2-wls-filter))

这给出了一个使用DepthAI的`rectified_right`和`depth`流进行主机端WLS过滤的示例。 

在 [BW1092](https://shop.luxonis.com/collections/all/products/bw1092-pre-order) 运行的示例如下所示:

![image](https://user-images.githubusercontent.com/32992551/94463964-fc920d00-017a-11eb-9e99-8a023cdc8a72.png)
