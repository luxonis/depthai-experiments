import depthai as dai
import numpy as np
try:
    while True:
        frame = node.inputs['frame_input'].get()
        detections_message = node.inputs['detections_input'].get()
        vehicle_detections = []
        
        for d in detections_message.detections:
            if d.label not in [2, 5, 7]: # not a car, bus or truck
                continue
            if d.confidence <  0.7:
                continue
            d.xmin = np.clip(d.xmin * 0.9, 0, 1)
            d.ymin = np.clip(d.ymin * 0.9, 0, 1)
            d.xmax = np.clip(d.xmax * 1.1, 0, 1)
            d.ymax = np.clip(d.ymax * 1.1, 0, 1)
            
            vehicle_detections.append(d)

            x_center = (d.xmin + d.xmax) / 2
            y_center = (d.ymin + d.ymax) / 2
            det_w = d.xmax - d.xmin
            det_h = d.ymax - d.ymin
                        
            det_center = dai.Point2f(x_center, y_center,normalized=True)
            det_size = dai.Size2f(det_w, det_h, normalized=True)
            
            det_rect = dai.RotatedRect(det_center, det_size, 0)
            det_rect = det_rect.denormalize(frame.getWidth(), frame.getHeight())
            
            cfg = dai.ImageManipConfigV2()
            cfg.addCropRotatedRect(det_rect, normalizedCoords=False)
            cfg.setOutputSize(640, 640)
            cfg.setReusePreviousImage(False)
            cfg.setTimestamp(detections_message.getTimestamp())
            
            node.outputs['output_config'].send(cfg)
            node.outputs['output_frame'].send(frame)
            
        vehicle_detections_msg = dai.ImgDetections()
        vehicle_detections_msg.detections = vehicle_detections
        vehicle_detections_msg.setTimestamp(detections_message.getTimestamp())
        vehicle_detections_msg.setTransformation(detections_message.getTransformation())
        
        node.outputs['output_vehicle_detections'].send(vehicle_detections_msg)
        
except Exception as e:
    node.warn(str(e))