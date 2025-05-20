import time
import depthai as dai

from utils.image import *
from utils.onnx import *
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

device_info = args.device
fps = args.fps_limit
ONNX_MODEL_PATH = args.model

print(f"depthai version: {dai.__version__}")

if args.resolution == 400:
    INFERENCE_H, INFERENCE_W = 416, 640
elif args.resolution == 800:
    INFERENCE_H, INFERENCE_W = 800, 1280
else:
    print("Invalid resolution, exiting.")
    exit(1)


def create_pipeline(device_info):
    def configure_cam(cam, size_x: int, size_y: int, fps: float):
        cap = dai.ImgFrameCapability()
        cap.size.fixed((size_x, size_y))
        cap.fps.fixed(fps)
        return cam.requestOutput(cap, True)

    if device_info is None: pipeline = dai.Pipeline()
    else: pipeline = dai.Pipeline(dai.Device(device_info))

    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

    # Linking
    if args.resolution == 800:
        monoLeftOut = configure_cam(monoLeft, 1280, 800, fps)
        monoRightOut = configure_cam(monoRight, 1280, 800, fps)
    else:
        monoLeftOut = configure_cam(monoLeft, 640, 400, fps)
        monoRightOut = configure_cam(monoRight, 640, 400, fps)

    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    queues = {
        "disp_queue": stereo.disparity.createOutputQueue(),
        "left_queue": stereo.rectifiedLeft.createOutputQueue(),
        "right_queue": stereo.rectifiedRight.createOutputQueue(),
    }

    return pipeline, queues

def get_device_and_pipeline(device_info=None):
    pipeline, queues = create_pipeline(device_info)
    device = pipeline.getDefaultDevice()
    pipeline.start()
    return device, pipeline, queues

if __name__ == "__main__":
    device, pipeline, queues = get_device_and_pipeline(device_info)
    with pipeline:
        device.setIrLaserDotProjectorIntensity(1)
        print(f"Connected device: {device.getDeviceName()}")

        onnx_session = load_onnx_model(ONNX_MODEL_PATH)

        print("Press F to generate Foundation Stereo Disparity")

        while pipeline.isRunning():
            disparity = queues["disp_queue"].get().getCvFrame()
            rectified_left = queues["left_queue"].get().getCvFrame()
            rectified_right = queues["right_queue"].get().getCvFrame()

            left = preprocess_image(rectified_left, (INFERENCE_H, INFERENCE_W))
            right = preprocess_image(rectified_right, (INFERENCE_H, INFERENCE_W))

            cv2.imshow("rectified_left", rectified_left)
            cv2.imshow("rectified_right", rectified_right)
            display_disparity(disparity, "disparity")

            key = cv2.waitKey(1)
            if key == ord('f') or key == ord('F'):
                print("Generating Foundation Stereo Disparity...")
                start = time.time()
                original_disp = preprocess_image(disparity, (INFERENCE_H, INFERENCE_W))
                original_disp_display = original_disp[0, 0]

                outputs = run_onnx_inference(onnx_session, left, right)
                ndr_disparity_display = outputs[0][0, 0]

                display_disparity(ndr_disparity_display, "ONNX Disparity", scale=255.0)
                display_disparity(original_disp_display, "Original Disparity", scale=255.0)
                end = time.time()
                print("Generated! Generation took:", round(end - start, 2), "seconds.")
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
