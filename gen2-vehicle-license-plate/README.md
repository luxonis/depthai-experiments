# Vehicle License Plate identification

This example demonstrates how to run a N stage inference on DepthAI using Gen2 Pipeline Builder.

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html).

This project is intended to work in the "Barrier" setting and aims to detect vehicle and license plates and save those detections. At a later point, this code may be extended to:

1/ Upload the saved detection image + data to a server

2/ Do more robust detecting



## Demo

TODO(wferrell): Update with an example video and screenshot.

[![License Plate Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Vehicle and license plate detection and identification on DepthAI")

## Pre-requisites

1. Purchase a DepthAI camera (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
main.py [-h] [-nd] [-cam] [-vid VIDEO] [-w WIDTH] [-lq]
```

Optional arguments:
 - `-h, --help` Show this help message and exit
 - `-nd, --no-debug` Prevent debug output
 - `-cam, --camera` Use DepthAI RGB camera for inference (conflicts with -vid)
 - `-vid VIDEO, --video VIDEO` Path to video file to be used for inference (conflicts with -cam)
 - `-w WIDTH, --width WIDTH` Visualization width
 - `-lq, --lowquality` Uses resized frames instead of source


To use with a video file, run the script with the following arguments

```
python3 main.py -vid input.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 


## Other links in Researching this

https://github.com/livezingy/license-plate-recognition/blob/master/testOpenVINO.py#L78
https://answers.opencv.org/question/205754/how-to-run-pretrained-model-with-openvino-on-rpi/
https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html


## Notes

export OPEN_MODEL_DOWNLOADER='/opt/intel//openvino_2021.2.185/deployment_tools/open_model_zoo/tools/downloader/downloader.py'


$OPEN_MODEL_DOWNLOADER --name vehicle-attributes-recognition-barrier-0039 --output_dir ~/open_model_zoo_downloads/


################|| Downloading vehicle-license-plate-detection-barrier-0106 ||################
        3. ################|| Downloading license-plate-recognition-barrier-0001 ||################
        4. ################|| Downloading vehicle-attributes-recognition-barrier-0039 ||################
        5.


/opt/intel//openvino_2021.2.185/deployment_tools/inference_engine/lib/intel64/myriad_compile


$MYRIAD_COMPILE -m ~/open_model_zoo_downloads/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4


$MYRIAD_COMPILE -m /Users/wferrell/open_model_zoo_downloads/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4


$MYRIAD_COMPILE -m /Users/wferrell/open_model_zoo_downloads/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml -ip U8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

$MYRIAD_COMPILE -m /Users/wferrell/open_model_zoo_downloads/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml -ip U8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4



$MYRIAD_COMPILE=/opt/intel/openvino_2021.2.185/deployment_tools/inference_engine/lib/intel64/myriad_compile
sudo install_name_tool -add_rpath $(dirname $(find /opt/intel/openvino_2021.2.185 -name libtbb.dylib)) $MYRIAD_COMPILE
sudo codesign --remove-signature $MYRIAD_COMPILE
