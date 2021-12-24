import time

l = [] # List of images
# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def remove_prev_frame(seq):
    for rm, frame in enumerate(l):
        if frame.getSequenceNum() == seq:
            # node.warn(f"List len {len(l)} Frame with same seq num: {rm},seq {seq}")
            break
    for i in range(rm):
        l.pop(0)

def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    time.sleep(0.001)
    preview = node.io['frame'].tryGet()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}")
        l.append(preview)

    face_dets = node.io['nn_in'].tryGet()
    # node.warn(f"Faces detected: {len(face_dets)}")
    if face_dets is not None:
        passthrough = node.io['passthrough'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue
        remove_prev_frame(seq)
        img = l[0] # Matching frame is the first in the list
        l.pop(0) # Remove matching frame from the list

        for det in face_dets.detections:
            limit_roi(det)
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(48, 96)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)