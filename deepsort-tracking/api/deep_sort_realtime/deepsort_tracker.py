import time
import logging

import cv2
import numpy as np

from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.utils.nms import non_max_suppression
# from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

logger = logging.getLogger(__name__)

EMBEDDER_CHOICES = [
    # "mobilenet",
    # "clip_RN50",
    # "clip_RN101",
    # "clip_RN50x4",
    # "clip_RN50x16",
    # "clip_ViT-B/32",
    # "clip_ViT-B/16",
]


class DeepSort(object):
    def __init__(
        self,
        max_age=30,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_wts=None,
        polygon=False,
        today=None,
    ):
        """

        Parameters
        ----------
        max_age : Optional[int] = 30
            Maximum number of missed misses before a track is deleted.
        nms_max_overlap : Optional[float] = 1.0
            Non-maxima suppression threshold: Maximum detection overlap, if is 1.0, nms will be disabled
        max_cosine_distance : Optional[float] = 0.2
            Gating threshold for cosine distance
        nn_budget :  Optional[int] = None
            Maximum size of the appearance descriptors, if None, no budget is enforced
        override_track_class : Optional[object] = None
            Giving this will override default Track class, this must inherit Track
        embedder : Optional[str] = 'mobilenet'
            Whether to use in-built embedder or not. If None, then embeddings must be given during update.
            Choice of ['mobilenet', 'clip_RN50', 'clip_RN101', 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
        half : Optional[bool] = True
            Whether to use half precision for deep embedder (applicable for mobilenet only)
        bgr : Optional[bool] = True
            Whether frame given to embedder is expected to be BGR or not (RGB)
        embedder_gpu: Optional[bool] = True
            Whether embedder uses gpu or not
        embedder_wts: Optional[str] = None
            Optional specification of path to embedder's model weights. Will default to looking for weights in `deep_sort_realtime/embedder/weights`. If deep_sort_realtime is installed as a package and CLIP models is used as embedder, best to provide path.
        polygon: Optional[bool] = False
            Whether detections are polygons (e.g. oriented bounding boxes)
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks
        """
        self.nms_max_overlap = nms_max_overlap
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(
            metric,
            max_age=max_age,
            # override_track_class=override_track_class,
            today=today,
        )

        if embedder is not None:
            if embedder not in EMBEDDER_CHOICES:
                raise Exception(f"Embedder {embedder} is not a valid choice.")
            # if embedder == "mobilenet":
            #     self.embedder = MobileNetv2_Embedder(embedder_wts, gpu=embedder_gpu, half=half, bgr=bgr)
        else:
            self.embedder = None
        self.polygon = polygon
        logger.info("DeepSort Tracker initialised")
        logger.info(f"- max age: {max_age}")
        logger.info(f"- appearance threshold: {max_cosine_distance}")
        logger.info(
            f'- nms threshold: {"OFF" if self.nms_max_overlap==1.0 else self.nms_max_overlap }'
        )
        logger.info(f"- max num of appearance features: {nn_budget}")
        logger.info(
            f'- overriding track class : {"No" if override_track_class is None else "Yes"}'
        )
        logger.info(f'- today given : {"No" if today is None else "Yes"}')
        logger.info(f'- in-build embedder : {"No" if self.embedder is None else "Yes"}')
        logger.info(f'- polygon detections : {"No" if polygon is False else "Yes"}')

    def update_tracks(self, raw_detections, embeds=None, frame=None, today=None):

        """Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        raw_detections (horizontal bb) : List[ Tuple[ List[float or int], float, str ] ]
            List of detections, each in tuples of ( [left,top,w,h] , confidence, detection_class)
        raw_detections (polygon) : List[ List[float], List[int or str], List[float] ]
            List of Polygons, Classes, Confidences. All 3 sublists of the same length. A polygon defined as a ndarray-like [x1,y1,x2,y2,...].
        embeds : Optional[ List[] ] = None
            List of appearance features corresponding to detections
        frame : Optional [ np.ndarray ] = None
            if embeds not given, Image frame must be given here, in [H,W,C].
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks

        Returns
        -------
        list of track objects (Look into track.py for more info or see "main" section below in this script to see simple example)

        """

        if embeds is None:
            if self.embedder is None:
                raise Exception(
                    "Embedder not created during init so embeddings must be given now!"
                )
            if frame is None:
                raise Exception("either embeddings or frame must be given!")

        if not self.polygon:
            # raw_detections = [d for d in raw_detections if d[0][2] > 0 and d[0][3] > 0]

            if embeds is None:
                embeds = self.generate_embeds(frame, raw_detections)

            # Proper deep sort detection objects that consist of bbox, confidence and embedding.
            detections = self.create_detections(raw_detections, embeds)
        else:
            polygons, bounding_rects = self.process_polygons(raw_detections[0])

            if embeds is None:
                embeds = self.generate_embeds_poly(frame, polygons, bounding_rects)

            # Proper deep sort detection objects that consist of bbox, confidence and embedding.
            detections = self.create_detections_poly(
                raw_detections, embeds, bounding_rects
            )

        # Run non-maxima suppression.
        boxes = np.array([d.ltwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # if self.nms_max_overlap < 1.0:
            # nms_tic = time.perf_counter()
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # nms_toc = time.perf_counter()
        # logger.debug(f'nms time: {nms_toc-nms_tic}s')
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections) # , today=today)

        return self.tracker.tracks

    def refresh_track_ids(self):
        self.tracker._next_id

    def generate_embeds(self, frame, raw_dets):
        crops = self.crop_bb(frame, raw_dets)
        return self.embedder.predict(crops)

    def generate_embeds_poly(self, frame, polygons, bounding_rects):
        crops = self.crop_poly_pad_black(frame, polygons, bounding_rects)
        return self.embedder.predict(crops)

    def create_detections(self, raw_dets, embeds):
        detection_list = []
        for i, (raw_det, embed) in enumerate(zip(raw_dets, embeds)):
            detection_list.append(
                Detection(raw_det[0], raw_det[1], embed, class_name=raw_det[2], id=i)
            )
        return detection_list

    def create_detections_poly(self, dets, embeds, bounding_rects):
        detection_list = []
        dets.extend([embeds, bounding_rects])
        for raw_polygon, cl, score, embed, bounding_rect in zip(*dets):
            x, y, w, h = bounding_rect
            x = max(0, x)
            y = max(0, y)
            bbox = [x, y, w, h]
            detection_list.append(
                Detection(bbox, score, embed, class_name=cl, others=raw_polygon)
            )
        return detection_list

    @staticmethod
    def process_polygons(raw_polygons):
        polygons = [
            [polygon[x : x + 2] for x in range(0, len(polygon), 2)]
            for polygon in raw_polygons
        ]
        bounding_rects = [
            cv2.boundingRect(np.array([polygon]).astype(int)) for polygon in polygons
        ]
        return polygons, bounding_rects

    @staticmethod
    def crop_bb(frame, raw_dets):
        crops = []
        im_height, im_width = frame.shape[:2]
        for detection in raw_dets:
            l, t, w, h = [int(x) for x in detection[0]]
            r = l + w
            b = t + h
            crop_l = max(0, l)
            crop_r = min(im_width, r)
            crop_t = max(0, t)
            crop_b = min(im_height, b)
            crops.append(frame[crop_t:crop_b, crop_l:crop_r])
        return crops

    @staticmethod
    def crop_poly_pad_black(frame, polygons, bounding_rects):
        masked_polys = []
        im_height, im_width = frame.shape[:2]
        for polygon, bounding_rect in zip(polygons, bounding_rects):
            mask = np.zeros(frame.shape, dtype=np.uint8)
            polygon_mask = np.array([polygon]).astype(int)
            cv2.fillPoly(mask, polygon_mask, color=(255, 255, 255))

            # apply the mask
            masked_image = cv2.bitwise_and(frame, mask)

            # crop masked image
            x, y, w, h = bounding_rect
            crop_l = max(0, x)
            crop_r = min(im_width, x + w)
            crop_t = max(0, y)
            crop_b = min(im_height, y + h)
            cropped = masked_image[crop_t:crop_b, crop_l:crop_r].copy()
            masked_polys.append(np.array(cropped))
        return masked_polys

    def delete_all_tracks(self):
        self.tracker.delete_all_tracks()
