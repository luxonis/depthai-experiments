from pathlib import Path

import cv2
import numpy as np
import depthai
import east

pipeline = depthai.Pipeline()

colorCam = pipeline.createColorCamera()
colorCam.setPreviewSize(256, 256)
colorCam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setCamId(0)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("preview")
colorCam.preview.link(cam_xout.input)

nn = pipeline.createNeuralNetwork()
nn.setBlobPath(str((Path(__file__).parent / Path('text-detection.blob')).resolve().absolute()))
nn.setNumPoolFrames(1)
colorCam.preview.link(nn.input)

nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("detections")
nn.out.link(nn_xout.input)

nn2 = pipeline.createNeuralNetwork()
nn2.setBlobPath(str((Path(__file__).parent / Path('text-recognition-0012.blob')).resolve().absolute()))

nn2_in = pipeline.createXLinkIn()
nn2_in.setStreamName("in_recognition")
nn2_in.out.link(nn2.input)

nn2_xout = pipeline.createXLinkOut()
nn2_xout.setStreamName("recognitions")
nn2.out.link(nn2_xout.input)

device = depthai.Device(pipeline)
device.startPipeline()

def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }

q_prev = device.getOutputQueue("preview")
# This should be blocking, but there's some sort of queuing!
q_det = device.getOutputQueue("detections", 1, overwrite=True)
q_rec_in = device.getInputQueue("in_recognition")
q_rec = device.getOutputQueue("recognitions")

frame = None
points = []
rec_pushed = 0
rec_received = 0

class CTCCodec(object):
    """ Convert between text-label and text-index """
    def __init__(self, characters):
        # characters (str): set of the possible characters.
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
             self.dict[char] = i + 1

    
        self.characters = dict_character
        #print(self.characters)
        #input()
        
    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    if self.characters[t[i]] != '#':
                        char_list.append(self.characters[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts

characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
codec = CTCCodec(characters)

while True:
    in_prev = q_prev.tryGet()
    in_rec = q_rec.tryGet()

    if in_prev is not None:
        shape = (3, in_prev.getHeight(), in_prev.getWidth())
        frame = in_prev.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_rec is not None:
        rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30,1,37)
        decoded_text = codec.decode(rec_data)
        print("=== ", rec_received, "text:", decoded_text)
        rec_received += 1

    if rec_received >= rec_pushed:
        in_det = q_det.tryGet()
        if in_det is not None:
            scores, geom1, geom2 = to_tensor_result(in_det).values()
            scores = np.reshape(scores, (1, 1, 64, 64))
            geom1 = np.reshape(geom1, (1, 4, 64, 64))
            geom2 = np.reshape(geom2, (1, 1, 64, 64))
    
            bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
            boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
            points = [
                east.rotated_Rectangle(bbox, angle * -1)
                for (bbox, angle) in zip(boxes, angles)
            ]

    if frame is not None:
        if in_det is not None:
            rec_received = 0
            rec_pushed = len(points)
            if rec_pushed:
                print("====== Pushing for recognition, count:", rec_pushed)
            cropped_stacked = None
            for point_arr in points:
                cv2.polylines(frame, [point_arr], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

                transformed = east.four_point_transform(frame, point_arr)
                transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
                transformed = cv2.resize(transformed, (120, 32), interpolation=cv2.INTER_AREA)
                transformed = np.ascontiguousarray(transformed)
                nn_data = depthai.NNData()
                nn_data.setLayer("Placeholder", transformed)
                q_rec_in.send(nn_data)
                if cropped_stacked is None:
                    cropped_stacked = transformed
                else:
                    cropped_stacked = np.vstack((cropped_stacked, transformed))
            if cropped_stacked is not None:
                cv2.imshow('cropped_stacked', cropped_stacked)
            in_det = None

            cv2.imshow('preview', frame)
            if cv2.waitKey(1) == ord('q'):
                break
