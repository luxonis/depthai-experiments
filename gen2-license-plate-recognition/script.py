import time

frames = [] # List of images
# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def remove_prev_frame(seq):
  found = False
  for rm, frame in enumerate(frames):
    if frame.getSequenceNum() == seq:
      ${debug}f"Detection matched frame with same seq num: {rm}, frame len: {len(frames)}, (seq {seq})")
      found = True
      break
  if not found:
    ${debug}f"No matching frame found! {seq}")
    raise Exception(f"No matching frame found! {seq}")
  for i in range(rm):
    frames.pop(0)

while True:
  frame_in = node.io['frame_in'].tryGet()
  if frame_in is not None:
    frames.append(frame_in)
    ${debug}f"New frame ({frame_in.getSequenceNum()}), frame len: {len(frames)}")
  else:
    time.sleep(0.001) # 1ms delay

  face_dets = node.io['det_in'].tryGet()
  if face_dets is not None:
    passthrough = node.io['passthrough'].get()
    seq = passthrough.getSequenceNum()
    remove_prev_frame(seq)
    ${debug}f"frame len: {len(frames)}")
    img = frames[0] # Matching frame is the first in the list
    frames.pop(0) # Remove matching frame from the list

    for det in face_dets.detections:
      cfg = ImageManipConfig()
      if det.xmin < 0: det.xmin = 0
      if det.ymin < 0: det.ymin = 0
      if det.xmax > 1: det.xmax = 1
      if det.ymax > 1: det.ymax = 1
      ${debug}f"Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
      cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
      cfg.setResize(${img_w}, ${img_h})
      cfg.setKeepAspectRatio(False)
      node.io['manip_cfg'].send(cfg)
      node.io['manip_frame'].send(img)