def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    face_dets = node.io['nn_in'].get().detections
    if len(face_dets) == 0: continue
    coords = face_dets[0] # take first
    
    coords.xmin -= 0.05
    coords.ymin -= 0.05
    coords.xmax += 0.05
    coords.ymax += 0.05
    
    limit_roi(coords)
    cfg = ImageManipConfig()
    cfg.setKeepAspectRatio(False)
    cfg.setCropRect(coords.xmin, coords.ymin, coords.xmax, coords.ymax)
    cfg.setResize(48, 48)
    node.io['to_manip'].send(cfg)