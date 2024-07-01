# coding=utf-8
from pathlib import Path
import argparse

import cv2
import depthai
import numpy as np
from imutils.video import FPS


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="prevent debug output"
)
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)",
)

parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="The path of the video file used for inference (conflicts with -cam)",
)

args = parser.parse_args()

debug = not args.no_debug


if args.camera and args.video:
    raise ValueError(
        'Command line parameter error! "-Cam" cannot be used together with "-vid"!'
    )
elif args.camera is False and args.video is None:
    raise ValueError(
        'Missing inference source! Use "-cam" to run on DepthAI cameras, or use "-vid <path>" to run on video files'
    )


def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()


def run_nn(x_in: depthai.InputQueue, x_out: depthai.MessageQueue, in_dict: dict[str, np.ndarray]) -> depthai.NNData:
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.addTensor(key, in_dict[key])
    x_in.send(nn_data)
    return x_out.tryGet()


class DepthAI:
    def __init__(
        self,
        file=None,
        camera=False,
    ):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 1 if self.camera else 2
        self.lineType = 0 if self.camera else 3


    def create_pipeline(self):
        print("Creating pipeline...")
        with depthai.Pipeline() as self.pipeline:
            if self.camera:
                # ColorCamera
                print("Creating Color Camera...")
                self.cam = self.pipeline.create(depthai.node.ColorCamera).build()
                self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
                self.cam.setResolution(
                    depthai.ColorCameraProperties.SensorResolution.THE_4_K
                )
                self.cam.setInterleaved(False)
                self.cam.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
                self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

                self.preview = self.cam.preview.createOutputQueue(maxSize=4, blocking=False)
                self.preview.setName("preview")

            self.create_nns()

            print("Pipeline created.")


    def create_nns(self):
        pass


    def create_nn(self, model_path: str, model_name: str, first: bool = False) -> tuple[depthai.MessageQueue, depthai.InputQueue | None]:
        """

        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.create(depthai.node.NeuralNetwork)
        model_nn.setBlobPath(str((Path(__file__).parent / Path(f"{model_path}")).resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first and self.camera:
            print("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in_q = model_nn.input.createInputQueue()

        model_out_q = model_nn.out.createOutputQueue(maxSize=4, blocking=False)
        model_out_q.setName(f"{model_name}_nn")
        return model_out_q, model_in_q


    def start_pipeline(self):
        print("Starting pipeline...")
        self.pipeline.start()


    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )


    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        self.parse_fun()

        if debug:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                print(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()


    def parse_fun(self):
        pass


    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

        cap.release()


    def run_camera(self):
        while self.pipeline.isRunning():
            in_rgb = self.preview.tryGet()
            if in_rgb is not None:
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break


    @property
    def cam_size(self):
        return self._cam_size


    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v


    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.cam_size = (255, 255)
        super().__init__(file, camera)


    def create_nns(self):
        self.fire_nn, self.fire_in = self.create_nn("models/fire-detection_openvino_2021.2_5shave.blob", "fire")


    def run_fire(self):
        labels = ["fire", "normal", "smoke"]
        w, h = self.frame.shape[:2]
        nn_data = run_nn(
            self.fire_in,
            self.fire_nn,
            {"Placeholder": to_planar(self.frame, (224, 224))},
        )
        if nn_data is None:
            return
        self.fps_nn.update()
        results = nn_data.getTensor("final_result").flatten()
        i = int(np.argmax(results))
        label = labels[i]
        if label == "normal":
            return
        else:
            if results[i] > 0.5:
                self.put_text(
                    f"{label}:{results[i]:.2f}",
                    (10, 25),
                    color=(0, 0, 255),
                    font_scale=1,
                )


    def parse_fun(self):
        self.run_fire()


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
