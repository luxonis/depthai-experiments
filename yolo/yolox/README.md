# YOLOX-tiny

More information about YOLOX can be found in the project's official repository: https://github.com/Megvii-BaseDetection/YOLOX

For more information about implementing YOLOX in DepthAI, including the drawbacks of the current approach, please see the discussion in the following issue: https://github.com/luxonis/depthai/issues/453

## Building a blob for a custom YOLOX model

1. Convert your model to OpenVINO by following these instructions: https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python
2. Compile the model to a blob for a myraid using the OpenVino model compiler, e.g.:
```shell
/opt/intel/openvino_2021/deployment_tools/tools/compile_tool/compile_tool -m yolox_tiny.xml -ip FP16 -d MYRIAD -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6
```