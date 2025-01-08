try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()
        
        #while dets.getTimestamp() > frame.getTimestamp():
        #    frame = node.inputs['preview'].get() 
        
        labels = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        only_one_detection = False
        score_threshold = 0.5

        for i, det in enumerate(dets.detections):
            if det.label not in labels:
                continue
            if det.confidence < score_threshold:
                continue
            cfg = ImageManipConfigV2()
            padding = 0.1
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.size.width = rect.size.width + padding * 2
            rect.size.height = rect.size.height + padding * 2
            rect.angle = 0

            cfg.addCropRotatedRect(rect=rect, normalizedCoords=True)
            cfg.setOutputSize(256, 256)

            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))