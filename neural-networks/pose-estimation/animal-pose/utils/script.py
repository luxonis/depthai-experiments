from typing import List

def generate_script_content(platform: str, resize_width: int = 224, resize_height: int = 224, padding: float = 0.1, valid_labels: List[int] = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]) -> str:
    """The function generates the script content for the dai.Script node. It is used to crop and resize the input image based on the detected object."""

    if platform.lower() == "rvc2":
        cfg_content = f"""
            cfg = ImageManipConfig()
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.size.width = rect.size.width + {padding} * 2
            rect.size.height = rect.size.height + {padding} * 2
            rect.angle = 0

            cfg.setCropRotatedRect(rect, True)
            cfg.setResize({resize_width}, {resize_height})
            cfg.setKeepAspectRatio(False)
        """
    elif platform.lower() == "rvc4":
        cfg_content = f"""
            cfg = ImageManipConfigV2()
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin
            rect.size.height = det.ymax - det.ymin
            rect.size.width = rect.size.width + {padding} * 2
            rect.size.height = rect.size.height + {padding} * 2
            rect.angle = 0
            cfg.addCropRotatedRect(rect=rect, normalizedCoords=True)
            cfg.setOutputSize({resize_width}, {resize_height})
        """
    else:
        raise ValueError("Unsupported version")

    return f"""
try:
    while True:
        frame = node.inputs['preview'].get()
        dets = node.inputs['det_in'].get()
        
        for i, det in enumerate(dets.detections):
            if det.label not in {valid_labels}:
                continue
            
            {cfg_content.strip()}
            node.outputs['manip_cfg'].send(cfg)
            node.outputs['manip_img'].send(frame)
except Exception as e:
    node.warn(str(e))
"""