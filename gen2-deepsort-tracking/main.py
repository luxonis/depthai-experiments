import cv2
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=1000, nn_budget=None, embedder=None, nms_max_overlap=1.0, max_cosine_distance=0.2)

def cb(packet: TwoStagePacket):
    detections = packet.img_detections.detections
    vis = packet.visualizer
    # Update the tracker
    object_tracks = tracker.iter(detections, packet.nnData, (640, 640))

    for track in object_tracks:
        if not track.is_confirmed() or \
            track.time_since_update > 1 or \
            track.detection_id >= len(detections) or \
            track.detection_id < 0:
            continue

        det = packet.detections[track.detection_id]
        vis.add_text(f'ID: {track.track_id}',
                        bbox=(*det.top_left, *det.bottom_right),
                        position=TextPosition.MID)
    frame = vis.draw(packet.frame)
    cv2.imshow('DeepSort tracker', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=15)
    yolo = oak.create_nn('yolov6nr3_coco_640x352', input=color)
    embedder = oak.create_nn('mobilenetv2_imagenet_embedder_224x224', input=yolo)

    oak.visualize(embedder, fps=True, callback=cb)
    # oak.show_graph()
    oak.start(blocking=True)

