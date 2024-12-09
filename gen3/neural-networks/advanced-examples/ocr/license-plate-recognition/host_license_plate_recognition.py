import depthai as dai
import numpy as np
import cv2

from detected_recognitions import DetectedRecognitions

PLATE_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    , "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>", "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>", "<Hainan>"
    , "<Hebei>", "<Heilongjiang>", "<Henan>", "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>", "<Jiangsu>"
    , "<Jiangxi>", "<Jilin>", "<Liaoning>", "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>", "<Shandong>", "<Shanghai>"
    , "<Shanxi>", "<Sichuan>", "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>", "<Zhejiang>", "<police>"
    , "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"
    , "W", "X", "Y", "Z"]

COLOR_LABELS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPE_LABELS = ["car", "bus", "truck", "van"]

COLOR = "tf.identity_1"
TYPE = "tf.identity"

class LicensePlateRecognition(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output
              , plate_images: dai.Node.Output
              , car_images: dai.Node.Output
              , plate_recognitions: dai.Node.Output
              , car_attributes: dai.Node.Output) -> "LicensePlateRecognition":
        self.link_args(preview
                       , plate_images
                       , car_images
                       , plate_recognitions
                       , car_attributes)
        self.sendProcessingToPipeline(True)
        return self

    # preview is actually type dai.ImgFrame here
    # plate_images, car_images, plate_recognitions and car_attributes are actually type DetectedRecognitions here
    def process(self, preview: dai.Buffer
                , plate_images: dai.Buffer
                , car_images: dai.Buffer
                , plate_recognitions: dai.Buffer
                , car_attributes: dai.Buffer) -> None:
        assert(isinstance(preview, dai.ImgFrame))
        assert(isinstance(plate_images, DetectedRecognitions))
        assert(isinstance(car_images, DetectedRecognitions))
        assert(isinstance(plate_recognitions, DetectedRecognitions))
        assert(isinstance(car_attributes, DetectedRecognitions))

        frame = preview.getCvFrame()
        stacked_plate_frame = None
        stacked_car_frame = None
        text_placeholder = np.zeros((72, 282, 3), np.uint8)

        if plate_images.data is not None and plate_recognitions.data is not None:
            for plate_frame, plate_data in zip(plate_images.data, plate_recognitions.data):
                plate_frame = plate_frame.getCvFrame()
                text_frame = text_placeholder.copy()

                plate_text = get_plate_from_nn(plate_data)
                cv2.putText(text_frame, plate_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX
                            , 0.5, (0, 255, 0))

                plate_frame = cv2.resize(plate_frame, (282, 72))
                stack_layer = np.hstack((plate_frame, text_frame))
                if stacked_plate_frame is None:
                    stacked_plate_frame = stack_layer
                else:
                    stacked_plate_frame = np.vstack((stacked_plate_frame, stack_layer))

            cv2.imshow("Recognized plates", stacked_plate_frame)


        if car_images.data is not None and car_attributes.data is not None:
            for car_frame, car_data in zip(car_images.data, car_attributes.data):
                text_frame = text_placeholder.copy()
                color, type, color_prob, type_prob = get_attributes_from_nn(car_data)
                cv2.putText(text_frame, color, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
                cv2.putText(text_frame, type, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
                cv2.putText(text_frame, f"{int(color_prob * 100)}%", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
                cv2.putText(text_frame, f"{int(type_prob * 100)}%", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

                stack_layer = np.hstack((car_frame.getCvFrame(), text_frame))
                if stacked_car_frame is None:
                    stacked_car_frame = stack_layer
                else:
                    stacked_car_frame = np.vstack((stacked_car_frame, stack_layer))

            cv2.imshow("Attributes", stacked_car_frame)

        for detection in car_images.img_detections.detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        for detection in plate_images.img_detections.detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()


def frame_norm(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def get_plate_from_nn(plate_data: dai.NNData) -> str:
    data = plate_data.getFirstTensor().flatten().astype(np.int32)
    plate_text = ""

    for idx in data:
        if idx == -1:
            break
        plate_text += PLATE_LABELS[idx]

    return plate_text

def get_attributes_from_nn(car_data: dai.NNData) -> tuple[str, str, float, float]:
    color = car_data.getTensor(COLOR).flatten()
    type = car_data.getTensor(TYPE).flatten()

    color_string = COLOR_LABELS[color.argmax()]
    type_string = CAR_TYPE_LABELS[type.argmax()]
    color_prob = float(color.max())
    type_prob = float(type.max())

    return color_string, type_string, color_prob, type_prob
