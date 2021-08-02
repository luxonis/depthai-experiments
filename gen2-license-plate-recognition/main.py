import argparse
import threading
import time
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np
from string import Template

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

# For script node debug
SCRIPT_DEBUG = False

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug
openvino_version = "2020.3"


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(frame, shape = None) -> np.ndarray:
    if shape is not None: frame = cv2.resize(frame, shape)
    return frame.transpose(2, 0, 1).flatten()

def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    return frame[0:height, delta:width-delta]

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2020_3)

    if args.camera:
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(1080, 1080)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # -----------------------------
    # Licence plate detection NN
    # -----------------------------
    # ImageManip that will crop the frame before sending it to the Licence plate detection NN
    lienseplate_det_manip = pipeline.create(dai.node.ImageManip)
    lienseplate_det_manip.initialConfig.setResize(300, 300)

    # NeuralNetwork
    print("Creating License Plates Detection Neural Network...")
    lic_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    lic_nn.setConfidenceThreshold(0.5)
    lic_nn.setBlobPath(str(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106", shaves=4, version=openvino_version)))
    lic_nn.input.setQueueSize(1)
    lic_nn.input.setBlocking(False)
    lienseplate_det_manip.out.link(lic_nn.input)

    lic_nn_xout = pipeline.create(dai.node.XLinkOut)
    lic_nn_xout.setStreamName("lic_nn_out")
    lic_nn.out.link(lic_nn_xout.input)

    # -----------------------------------
    # Vehicle detection NN
    # -----------------------------------
    # ImageManip that will crop the frame before sending it to the Vehicle detection NN
    vehicle_det_manip = pipeline.create(dai.node.ImageManip)
    vehicle_det_manip.initialConfig.setResize(672, 384)

    # NeuralNetwork
    print("Creating Vehicle Detection Neural Network...")
    veh_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    veh_nn.setConfidenceThreshold(0.5)
    veh_nn.setBlobPath(str(blobconverter.from_zoo(name="vehicle-detection-adas-0002", shaves=4, version=openvino_version)))
    veh_nn.input.setQueueSize(1)
    veh_nn.input.setBlocking(False)
    vehicle_det_manip.out.link(veh_nn.input)

    # Send vehicle detections to the host (for BBs)
    veh_nn_xout = pipeline.create(dai.node.XLinkOut)
    veh_nn_xout.setStreamName("veh_nn_out")
    veh_nn.out.link(veh_nn_xout.input)

    # -----------------------------------
    # Script nodes
    # -----------------------------------
    with open('script.py', 'r') as file:
        script_template = Template(file.read())
    
    script_lic = pipeline.create(dai.node.Script)
    script_lic.setScript(script_template.substitute(
        debug = 'node.warn("LIC] " + ' if SCRIPT_DEBUG else '#',
        img_w = '94',
        img_h = '24'
    ))
    lic_nn.out.link(script_lic.inputs['det_in'])
    lic_nn.passthrough.link(script_lic.inputs['passthrough'])

    script_veh = pipeline.create(dai.node.Script)
    script_veh.setScript(script_template.substitute(
        debug = 'node.warn("VEH] " + ' if SCRIPT_DEBUG else '#',
        img_w = '72',
        img_h = '72'
    ))
    veh_nn.out.link(script_veh.inputs['det_in'])
    veh_nn.passthrough.link(script_veh.inputs['passthrough'])
    # -----------------------------------
    # Licence plate recognition NN
    # -----------------------------------
    lic_rec_manip = pipeline.create(dai.node.ImageManip)
    lic_rec_manip.initialConfig.setResize(94, 24)
    lic_rec_manip.setWaitForConfigInput(False)
    script_lic.outputs['manip_cfg'].link(lic_rec_manip.inputConfig)
    script_lic.outputs['manip_frame'].link(lic_rec_manip.inputImage)

    rec_nn = pipeline.create(dai.node.NeuralNetwork)
    rec_nn.setBlobPath(str(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007", shaves=4, version=openvino_version)))
    lic_rec_manip.out.link(rec_nn.input)

    rec_xout = pipeline.create(dai.node.XLinkOut)
    rec_xout.setStreamName("rec_nn")
    rec_nn.out.link(rec_xout.input)

    rec_pass = pipeline.create(dai.node.XLinkOut)
    rec_pass.setStreamName("rec_pass")
    rec_nn.passthrough.link(rec_pass.input)

    # -----------------------------------
    # Vehicle attributes recognition NN
    # -----------------------------------

    veh_rec_manip = pipeline.create(dai.node.ImageManip)
    veh_rec_manip.initialConfig.setResize(72, 72)
    veh_rec_manip.setWaitForConfigInput(False)
    script_veh.outputs['manip_cfg'].link(veh_rec_manip.inputConfig)
    script_veh.outputs['manip_frame'].link(veh_rec_manip.inputImage)

    attr_nn = pipeline.create(dai.node.NeuralNetwork)
    attr_nn.setBlobPath(str(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039", shaves=4, version=openvino_version)))
    veh_rec_manip.out.link(attr_nn.input)

    attr_xout = pipeline.create(dai.node.XLinkOut)
    attr_xout.setStreamName("attr_nn")
    attr_nn.out.link(attr_xout.input)

    attr_pass = pipeline.create(dai.node.XLinkOut)
    attr_pass.setStreamName("attr_pass")
    attr_nn.passthrough.link(attr_pass.input)

    if args.camera:
        cam.preview.link(lienseplate_det_manip.inputImage)
        cam.preview.link(vehicle_det_manip.inputImage)
        cam.preview.link(script_lic.inputs['frame_in'])
        # cam.preview.link(script_veh.inputs['frame_in'])
        vehicle_det_manip.out.link(script_veh.inputs['frame_in'])
    else:
        frame_xin = pipeline.create(dai.node.XLinkIn)
        frame_xin.setStreamName("video_in")

        frame_xin.out.link(lienseplate_det_manip.inputImage)
        frame_xin.out.link(vehicle_det_manip.inputImage)
        frame_xin.out.link(script_lic.inputs['frame_in'])
        frame_xin.out.link(script_veh.inputs['frame_in'])

    print("Pipeline created.")
    return pipeline


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


running = True
license_detections = []
vehicle_detections = []
rec_results = []
attr_results = []
frame_det_seq = 0 # For video input

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)

def rec_thread(q_rec, q_pass):
    items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>", "<Gansu>",
            "<Guangdong>", "<Guangxi>", "<Guizhou>", "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>", "<HongKong>",
            "<Hubei>", "<Hunan>", "<InnerMongolia>", "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>", "<Macau>",
            "<Ningxia>", "<Qinghai>", "<Shaanxi>", "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>", "<Tianjin>",
            "<Tibet>", "<Xinjiang>", "<Yunnan>", "<Zhejiang>", "<police>", "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    global rec_results

    while running:
        try:
            rec_data = q_rec.get().getFirstLayerFp16()
            rec_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        decoded_text = ""
        for idx in rec_data:
            if idx == -1:
                break
            decoded_text += items[int(idx)]
        rec_results = [(cv2.resize(rec_frame, (200, 64)), decoded_text)] + rec_results[:9]
        fps.tick_fps('rec')

def attr_thread(q_attr, q_pass):
    global attr_results

    while running:
        try:
            attr_data = q_attr.get()
            attr_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        types = ["car", "bus", "truck", "van"]

        in_color = np.array(attr_data.getLayerFp16("color"))
        in_type = np.array(attr_data.getLayerFp16("type"))

        color = colors[in_color.argmax()]
        color_prob = float(in_color.max())
        type = types[in_type.argmax()]
        type_prob = float(in_type.max())

        attr_results = [(attr_frame, color, type, color_prob, type_prob)] + attr_results[:9]

        fps.tick_fps('attr')

print("Starting pipeline...")
with dai.Device(create_pipeline()) as device:
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        video_in = device.getInputQueue("video_in")

    lic_nn_q = device.getOutputQueue("lic_nn_out", 1, False)
    veh_nn_q = device.getOutputQueue("veh_nn_out", 1, False)

    rec_nn = device.getOutputQueue("rec_nn", 1, False)
    rec_pass = device.getOutputQueue("rec_pass", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)

    rec_t = threading.Thread(target=rec_thread, args=(rec_nn, rec_pass))
    rec_t.start()
    attr_t = threading.Thread(target=attr_thread, args=(attr_nn, attr_pass))
    attr_t.start()


    def should_run():
        return cap.isOpened() if args.video else True

    def get_frame():
        if args.video:
            return cap.read()
        else:
            return True, cam_out.get().getCvFrame()

    def draw_rect(frame, detections, color):
        for detection in detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            fps.next_iter()

            if not args.camera:
                frame = crop_to_square(frame)
                imgFrame = dai.ImgFrame()
                imgFrame.setData(to_planar(frame))
                imgFrame.setSequenceNum(frame_det_seq)
                imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
                h, w, c = frame.shape
                imgFrame.setWidth(w)
                imgFrame.setHeight(h)
                video_in.send(imgFrame)
                frame_det_seq += 1

            lic_in = lic_nn_q.tryGet()
            veh_in = veh_nn_q.tryGet()
            
            if debug:
                debug_frame = frame.copy()

                if veh_in is not None:
                    fps.tick_fps('veh')
                    draw_rect(debug_frame, veh_in.detections, (200,20,20))

                if lic_in is not None:
                    fps.tick_fps('lic')
                    draw_rect(debug_frame, lic_in.detections, (10,200,200))

                for detection in license_detections + vehicle_detections:
                    bbox = frame_norm(debug_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0))
                cv2.putText(debug_frame, f"LIC FPS:  {round(fps.tick_fps('lic'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"VEH FPS:  {round(fps.tick_fps('veh'), 1)}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"REC FPS:  {round(fps.tick_fps('rec'), 1)}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"ATTR FPS:  {round(fps.tick_fps('attr'), 1)}", (5, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.imshow("rgb", debug_frame)

                rec_stacked = None

                for rec_img, rec_text in rec_results:
                    rec_placeholder_img = np.zeros((64, 200, 3), np.uint8)
                    cv2.putText(rec_placeholder_img, rec_text, (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    rec_combined = np.hstack((rec_img, rec_placeholder_img))

                    if rec_stacked is None:
                        rec_stacked = rec_combined
                    else:
                        rec_stacked = np.vstack((rec_stacked, rec_combined))

                if rec_stacked is not None:
                    cv2.imshow("Recognized plates", rec_stacked)

                attr_stacked = None

                for attr_img, attr_color, attr_type, color_prob, type_prob in attr_results:
                    attr_placeholder_img = np.zeros((72, 200, 3), np.uint8)
                    cv2.putText(attr_placeholder_img, attr_color, (15, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, attr_type, (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(color_prob * 100)}%", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(type_prob * 100)}%", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    attr_combined = np.hstack((attr_img, attr_placeholder_img))

                    if attr_stacked is None:
                        attr_stacked = attr_combined
                    else:
                        attr_stacked = np.vstack((attr_stacked, attr_combined))

                if attr_stacked is not None:
                    cv2.imshow("Attributes", attr_stacked)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    running = False

rec_t.join()
attr_t.join()
print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
