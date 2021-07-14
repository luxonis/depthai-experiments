from pathlib import Path
from imutils.video import FPS
from .utils import *

class DepthAI:

    def __init__(self,file=None,camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()
        self.fps = FPS()
        self.fps.start()

    
    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()
        if self.camera:
            print("Creating Color Camera...")
            self.cam = self.pipeline.createColorCamera()
            self.cam.setPreviewSize(self._cam_size[0],self._cam_size[1])
            self.cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            self.cam.setInterleaved(False)
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            self.cam.preview.link(cam_xout.input)
        
        self.create_nns()
    
    def create_nns(self):
        pass
    
    def create_models(self, model_path,model_name,first_model=False):
        print(f"Start creating{model_path}Neural Networks")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        if first_model and self.camera:
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(self,model_path,model_name,first_model=False,conf=0.5):
        print(f"Start creating{model_path}Neural Networks")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)
        if first_model and self.camera:
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)
    
    def create_yolo_nn(self,model_path,model_name,first_model=False):
        print(f"Start creating{model_path}Neural Networks")
        model_nn = self.pipeline.createYoloDetectionNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)
        if first_model:
            if self.camera:
                self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")
        self.device.startPipeline()

        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 4, False)
        
        self.start_nns()
    
    def start_nns(self):
        pass

    def full_frame_cords(self, cords):
        original_cords = self.face_coords[0]
        return [
            original_cords[0 if i % 2 == 0 else 1] + val
            for i, val in enumerate(cords)
        ]

    def full_frame_bbox(self, bbox):
        relative_cords = self.full_frame_cords(bbox)
        height, width = self.frame.shape[:2]
        y_min = max(0, relative_cords[1])
        y_max = min(height, relative_cords[3])
        x_min = max(0, relative_cords[0])
        x_max = min(width, relative_cords[2])
        result_frame = self.frame[y_min:y_max, x_min:x_max]
        return result_frame, relative_cords


    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

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
        while True:
            in_rgb = self.cam_out.tryGet()
            if in_rgb is not None:
                shape = (3,in_rgb.getHeight(),in_rgb.getWidth())
                self.frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break
    
    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()
        self.parse_run()
        if debug:
            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, ( int(900),  int(900 / aspect_ratio))))
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration()
    
    def parse_run(self):
        pass
    
    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self,v):
        self._cam_size = v
    
    @property
    def get_cam_size(self):
        return (self._cam_size[0],self._cam_size[1])

    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        self.fps.stop()
        print("FPS:{:.2f}".format(self.fps.fps()))
        del self.device


