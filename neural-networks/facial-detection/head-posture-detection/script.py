import time


msgs = dict()


def add_msg(msg, name, seq=None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)

    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

    if 15 < len(msgs):
        node.warn(f"Removing first element! len {len(msgs)}")
        msgs.popitem()


def get_msgs():
    global msgs
    seq_remove = []
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq)

        if len(syncMsgs) == 2:
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs
    return None


def correct_bb(xmin, ymin, xmax, ymax):
    if xmin < 0:
        xmin = 0.001
    if ymin < 0:
        ymin = 0.001
    if xmax > 1:
        xmax = 0.999
    if ymax > 1:
        ymax = 0.999
    return [xmin, ymin, xmax, ymax]


while True:
    time.sleep(0.001)  # Avoid lazy looping

    preview = node.io["preview"].tryGet()
    if preview is not None:
        add_msg(preview, "preview")

    face_dets = node.io["face_det_in"].tryGet()
    if face_dets is not None:
        seq = face_dets.getSequenceNum()
        add_msg(face_dets, "dets", seq)

    sync_msgs = get_msgs()
    if sync_msgs is not None:
        img = sync_msgs["preview"]
        dets = sync_msgs["dets"]
        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfig()
            bb = correct_bb(
                det.xmin - 0.03, det.ymin - 0.03, det.xmax + 0.03, det.ymax + 0.03
            )
            cfg.setCropRect(*bb)
            cfg.setResize(60, 60)
            cfg.setKeepAspectRatio(False)
            node.io["manip_cfg"].send(cfg)
            node.io["manip_img"].send(img)
