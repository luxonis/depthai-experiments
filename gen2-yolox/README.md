# YOLOX-tiny using the Gen2 API

More information about YOLOX can be found in the project's official repository: https://github.com/Megvii-BaseDetection/YOLOX

For more information about implementing YOLOX in DepthAi, including the drawbacks of the current approach, please see the discussion in the following issue: https://github.com/luxonis/depthai/issues/453

## Building a blob for a custom YOLOX model

1. Download YoloX model in OpenVINO IR format - https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python#download-openvino-models
2. Compile the model to a MyriadX Blob using

   1. OpenVino model compiler
     ```shell
     /opt/intel/openvino_2021/deployment_tools/tools/compile_tool/compile_tool -m yolox_tiny.xml -ip FP16 -d MYRIAD -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6
     ```
   2. BlobConverter
     ```shell
     python3 -m pip install blobconverter
     python3 -m blobconverter --openvino-xml yolo_tiny.xml --openvino-bin yolo_tiny.bin --shaves 6 --compile-params 
     ```
   