try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()
        
        while dets.getTimestamp() > frame.getTimestamp():
            frame = node.inputs['preview'].get() 
        
        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfigV2()
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.angle = 0
            cfg.addCropRotatedRect(rect=rect, normalizedCoords=True)
            cfg.addResize(256, 256)

            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))