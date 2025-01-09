try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()

        labels = [56] # chair
        padding = 0.2
        
        for i, det in enumerate(dets.detections):
            if det.label not in labels:
                continue
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin - padding, det.ymin - padding, det.xmax + padding, det.ymax + padding)
            cfg.setResize(224, 224)
            cfg.setKeepAspectRatio(False)

            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)

except Exception as e:
    node.warn(str(e))