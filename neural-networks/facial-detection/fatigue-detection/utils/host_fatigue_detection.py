import depthai as dai
from collections import deque
from utils.face_landmarks import determine_fatigue

class FatigueDetection(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        
        self.crop_face  =self.createInput()
        self.preview = self.createInput()
        self.face_nn = self.createInput()
        self.landmarks_nn = self.createInput()
        
        self.closed_eye_duration = deque(maxlen=30)
        self.head_tilted_duration = deque(maxlen=30)
        
        self.out = self.createOutput()
        self.output_frame = self.createOutput()
    
    def create_output_text(self, text: str, x_position: float, y_position: float):
        text_annotation = dai.TextAnnotation()
        text_annotation.position = dai.Point2f(x_position, y_position)
        text_annotation.text = text
        text_annotation.textColor = dai.Color(0.5, 0.5, 1.0, 1.0)
        text_annotation.fontSize = 45
        text_annotation.backgroundColor = dai.Color(1.0, 1.0, 0.5, 1.0)
        
        return text_annotation
    
    def run(self) -> None:
        while self.isRunning():
            frame = self.preview.get().getCvFrame()
            face_dets = self.face_nn.get()
            dets = face_dets.detections
            n_detections = len(dets)
            
            img_annotations = dai.ImgAnnotations()
            annotation = dai.ImgAnnotation()
            
            if n_detections >= 1:
                crop_face = self.crop_face.get().getCvFrame()
                landmarks = self.landmarks_nn.get()
                pitch, eyes_closed= determine_fatigue(frame, landmarks)
                
                for i in range(n_detections - 1): # skip the rest of the detections
                    self.crop_face.get().getCvFrame()
                    self.landmarks_nn.get()

                self.head_tilted_duration.append(pitch)
                self.closed_eye_duration.append(eyes_closed)
                
                percent_closed_eyes = (sum(self.closed_eye_duration) / len(self.closed_eye_duration))
                percent_tilted = (sum(self.head_tilted_duration) / len(self.head_tilted_duration))
                
                if percent_tilted >= 0.75:
                    annotation.texts.append(self.create_output_text("Head Tilted!", 0.1, 0.1))
                    
                if percent_closed_eyes >= 0.75:
                    annotation.texts.append(self.create_output_text("Eyes Closed!", 0.1, 0.2))
                
            
            img_annotations.annotations.append(annotation)
            img_annotations.setTimestamp(face_dets.getTimestamp())
            self.out.send(img_annotations)
