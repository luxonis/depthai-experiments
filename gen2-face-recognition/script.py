import time
bboxes = [] # List of face BBs
l = [] # List of images

# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def remove_prev_frames(seq):
    for rm, frame in enumerate(l):
        if frame.getSequenceNum() == seq:
            # node.warn(f"List len {len(l)} Frame with same seq num: {rm},seq {seq}")
            break
    for i in range(rm):
        l.pop(0)

def find_frame(seq):
    for frame in l:
        if frame.getSequenceNum() == seq:
            return frame

while True:
    preview = node.io['preview'].get()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}, size {len(l)}")
        l.append(preview)
        # node.warn(f'Imgs len {len(l)}')
        if 13 == len(l):
            node.io['host_frame'].send(l.pop(0))


    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        passthrough = node.io['face_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue

        # node.warn(f"FACE_DET size {len(l)}")
        img = find_frame(seq) # Matching frame is the first in the list
        node.warn(str(img))
        for det in face_dets.detections:
            bboxes.append(det) # For the rotation
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(60, 60)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)

    headpose = node.io['headpose_in'].tryGet()
    if headpose is not None:
        passthrough = node.io['headpose_pass'].get()
        seq = passthrough.getSequenceNum()
        # Face rotation in degrees
        r = headpose.getLayerFp16('angle_r_fc')[0] # Only 1 float in there
        bb = bboxes.pop(0) # Get BB from the img detection

        # remove_prev_frame(seq)
        remove_prev_frames(seq)
        img = l.pop(0) # Matching frame is the first in the list
        node.warn('HP' + str(img))
        node.warn('bb' + str(bb))
        node.io['host_frame'].send(img)
        cfg = ImageManipConfig()
        rr = RotatedRect()
        rr.center.x = (bb.xmin + bb.xmax) / 2
        rr.center.y = (bb.ymin + bb.ymax) / 2
        rr.size.width = bb.xmax - bb.xmin
        rr.size.height = bb.ymax - bb.ymin
        rr.angle = r # Rotate the rect in opposite direction
        # True = coordinates are normalized (0..1)
        cfg.setCropRotatedRect(rr, True)
        cfg.setResize(112, 112)
        cfg.setKeepAspectRatio(False)

        node.io['manip2_cfg'].send(cfg)
        node.io['manip2_img'].send(img)