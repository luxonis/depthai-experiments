# try:
#     while True:
#         frame = node.inputs['preview'].get()
#         dets = node.inputs['det_in'].get()

#         labels = [56] # chair
#         only_one_detection = False
#         score_threshold = 0.5
#         padding = 0.2

#         for i, det in enumerate(dets.detections):
#             cfg = ImageManipConfig()
#             rect = RotatedRect()
#             rect.center.x = (det.xmin + det.xmax) / 2
#             rect.center.y = (det.ymin + det.ymax) / 2
#             rect.size.width = det.xmax - det.xmin
#             rect.size.height = det.ymax - det.ymin
#             rect.size.width = rect.size.width + padding * 2
#             rect.size.height = rect.size.height + padding * 2
#             rect.angle = 0

#             cfg.setCropRotatedRect(rect, True)
#             cfg.setResize(224, 224)
#             cfg.setKeepAspectRatio(False)
#             node.outputs['manip_cfg'].send(cfg)
#             node.outputs['manip_img'].send(frame)
#             break

# except Exception as e:
#     node.warn(str(e))

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

    detections = node.io['det_in'].tryGet()
    if detections is not None:
        seq = detections.getSequenceNum()
        add_msg(detections, 'dets', seq)

    sync_msgs = get_msgs()

    labels = [56] # chair
    score_threshold = 0.5
    padding = 0.2

    if sync_msgs is not None:
        img = sync_msgs['preview']
        dets = sync_msgs['dets']
        for i, det in enumerate(dets.detections):
            if det.label not in labels:
                continue
            if det.confidence < score_threshold:
                continue
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin - padding, det.ymin - padding, det.xmax + padding, det.ymax + padding)
            cfg.setResize(224, 224)
            cfg.setKeepAspectRatio(False)
            
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(img)