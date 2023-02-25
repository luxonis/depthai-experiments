import sys

# enlarge crop area for a given factor
def enlarge_crop_area(xmin, ymin, xmax, ymax, factor=1.2):

    # current size of the image and the crop
    img_w, img_h = 1.0, 1.0
    crop_w, crop_h = xmax - xmin, ymax - ymin

    # compute the new size
    size_new = max([crop_w, crop_h]) * factor
    if size_new > min([img_w, img_h]):
        size_new = min([img_w, img_h])

    # compute the shift
    shift_x = (size_new - crop_w) / 2
    shift_y = (size_new - crop_h) / 2

    xmin -= shift_x
    ymin -= shift_y

    if xmin < 0:
        xmin = 0

    if ymin < 0:
        ymin = 0

    # node.warn("line ??")
    xmax = xmin + size_new
    ymax = ymin + size_new

    if xmax > img_w:
        xmax = img_w

    if ymax > img_h:
        ymax = img_h

    return xmin, ymin, xmax, ymax

l = [] # List of assets
# The correct frame will be the first in the list
def remove_prev_frame(seq):
    for rm, frame in enumerate(l):
        if frame.getSequenceNum() == seq:
            break
    for i in range(rm):
        l.pop(0)

while True:
    preview = node.io['preview'].tryGet()
    if preview is not None:
        l.append(preview)

    det_in = node.io['det_in'].tryGet()
    if det_in is not None:
        passthrough = node.io['det_passthrough'].get()
        seq = passthrough.getSequenceNum()

        if len(l) == 0:
            continue

        remove_prev_frame(seq)
        img = l[0] # Matching frame is the first in the list

        l.pop(0) # Remove matching frame from the list

        detections = det_in.detections
        if detections:
            # find the detection with highest confidence
            # label of 9 indicates chair in VOC
            confs = [det.confidence for det in detections if det.label == 9]
            # confs = [det.confidence for det in detections if det.label == 62]
            if len(confs) > 0:
                idx = confs.index(max(confs))
                detection = detections[idx]

                x1 = detection.xmax # top left
                y1 = detection.ymin # top left
                x2 = detection.xmin # bottom right
                y2 = detection.ymax # bottom right

                cfg = ImageManipConfig()

                xmin, ymin, xmax, ymax = enlarge_crop_area(x1, y1, x2, y2)

                cfg.setCropRect(xmin, ymin, xmax, ymax)
                cfg.setResize(224, 224)
                cfg.setKeepAspectRatio(True)
                node.io['manip_img'].send(img)
                node.io['manip_cfg'].send(cfg)
