from typing import List


def generate_script_content(
    resize_width: int = 192,
    resize_height: int = 256,
    padding: float = 0.1,
    valid_labels: List[int] = [0],
) -> str:
    """The function generates the script content for the dai.Script node. It is used to crop and resize the input image based on the detected object."""

    return f"""
try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()
        
        for i, det in enumerate(dets.detections):
            if det.label not in {valid_labels}:
                continue
            
            cfg = ImageManipConfigV2()
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.size.width = rect.size.width + {padding} * 2
            rect.size.height = rect.size.height + {padding} * 2
            rect.angle = 0
            cfg.addCropRotatedRect(rect, True)
            cfg.setOutputSize({resize_width}, {resize_height}, ImageManipConfigV2.ResizeMode.STRETCH)

            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)
except Exception as e:
    node.warn(str(e))
"""
