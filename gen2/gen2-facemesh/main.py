from depthai_sdk import OakCamera, Visualizer
from depthai_sdk.classes.packets import TwoStagePacket
from depthai_sdk.classes.nn_results import ImgLandmarks
from depthai_sdk.visualize.bbox import BoundingBox
import cv2
import depthai as dai

# Confidence threshold for the facemesh model
THRESHOLD = 0.3
# 5% padding for face detection, as facemesh_192x192 is trained on the whole head not just the face
PADDING = 5

# We will be saving the passthrough frames so we can draw landmarks on it
pass_f = None
def pass_cb(packet):
    global pass_f
    pass_f = packet.frame

def draw_rect(frame, color, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, color, 1)

def cb(packet: TwoStagePacket):
    global pass_f
    vis: Visualizer = packet.visualizer
    vis.draw(packet.frame)
    frame_full = packet.frame

    # face-detection-retail-0004 is 1:1 aspect ratio
    pre_det_crop_bb = BoundingBox().resize_to_aspect_ratio(frame_full.shape, (1, 1), resize_mode='crop')
    # draw_rect(frame_full, (255, 127, 0), *pre_det_crop_bb.denormalize(frame_full.shape))

    for det, imgLdms in zip(packet.detections, packet.nnData):
        if imgLdms is None or imgLdms.landmarks is None:
            continue
        imgLdms: ImgLandmarks

        img_det: dai.ImgDetection = det.img_detection
        det_bb = pre_det_crop_bb.get_relative_bbox(BoundingBox(img_det))
        # draw_rect(frame_full, (255, 0, 0), *det_bb.denormalize(frame_full.shape))

        padding_bb = det_bb.add_padding(0.05, pre_det_crop_bb)
        draw_rect(frame_full, (0, 0, 255), *padding_bb.denormalize(frame_full.shape))
        for ldm, clr in zip(imgLdms.landmarks, imgLdms.colors):
            mapped_ldm = padding_bb.map_point(*ldm).denormalize(frame_full.shape)
            cv2.circle(frame_full, center=mapped_ldm, radius=1, color=clr, thickness=-1)

            if pass_f is not None:
                cv2.circle(pass_f, center=(int(ldm[0]*192), int(ldm[1]*192)), radius=1, color=clr, thickness=-1)

    cv2.imshow('Facemesh', frame_full)
    if pass_f is not None:
        cv2.imshow('Passthrough', cv2.pyrUp(pass_f))

with OakCamera() as oak:
    color = oak.create_camera('color')

    det_nn = oak.create_nn('face-detection-retail-0004', color)
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det_nn.config_nn(resize_mode='crop')

    facemesh_nn = oak.create_nn('facemesh_192x192', input=det_nn)
    facemesh_nn.config_multistage_nn(scale_bb=(5,5))

    # Send the 2stage NN results to the callback
    oak.visualize(facemesh_nn, callback=cb).detections(fill_transparency=0)
    # Send the crops to the passthrough callback
    oak.callback(facemesh_nn.out.twostage_crops, pass_cb)
    oak.visualize(det_nn.out.passthrough)

    # oak.show_graph() # Show pipeline graph
    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)
