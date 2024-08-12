msgs = dict()


def add_msg(msg, name, seq = None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)

    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg


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


def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb


while True:
    preview = node.io['preview'].tryGet()
    if preview is not None:
        add_msg(preview, 'preview')

    detections = node.io['detections'].tryGet()
    if detections is not None:
        seq = detections.getSequenceNum()
        add_msg(detections, 'dets', seq)

    sync_msgs = get_msgs()
    if sync_msgs is not None:
        img = sync_msgs['preview']
        dets = sync_msgs['dets']
        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(224, 224) 
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)