import time
sync = {} # Dict of messages

# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def get_sync(target_seq):
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, msgs in sync.items():
        if seq == str(target_seq):
            # We have synced msgs, remove previous msgs (memory cleaning)
            for rm in seq_remove:
                del sync[rm]
            return msgs
        seq_remove.append(seq) # Will get removed from dict if we find synced sync pair
    return None
def find_frame(target_seq):
    if str(target_seq) in sync and "frame" in sync[str(target_seq)]:
        return sync[str(target_seq)]["frame"]
def add_detections(det, seq):
    # No detections, we can remove saved frame
    if len(det) == 0:
        del sync[str(seq)]
    else:
        # Save detections, as we will need them for face recognition model
        sync[str(seq)]["detections"] = det

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999

while True:
    time.sleep(0.001)
    preview = node.io['preview'].tryGet()
    if preview is not None:
        sync[str(preview.getSequenceNum())] = {}
        sync[str(preview.getSequenceNum())]["frame"] = preview

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        # node.warn(f"New detection start")
        passthrough = node.io['face_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(sync) == 0: continue
        img = find_frame(seq) # Matching frame is the first in the list
        if img is None: continue

        add_detections(face_dets.detections, seq)

        for det in face_dets.detections:
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(60, 60)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)

    headpose = node.io['headpose_in'].tryGet()
    if headpose is not None:
        # node.warn(f"New headpose")
        passthrough = node.io['headpose_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New headpose seq {seq}")
        # Face rotation in degrees
        r = headpose.getLayerFp16('angle_r_fc')[0] # Only 1 float in there

        msgs = get_sync(seq)
        bb = msgs["detections"].pop(0)
        correct_bb(bb)

        # remove_prev_frame(seq)
        img = msgs["frame"]
        # node.warn('HP' + str(img))
        # node.warn('bb' + str(bb))
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
        cfg.setKeepAspectRatio(True)

        node.io['manip2_cfg'].send(cfg)
        node.io['manip2_img'].send(img)