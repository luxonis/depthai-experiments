import cv2
import depthai as dai
import numpy as np
from pyzbar.pyzbar import decode

DECODE = True


class QRScanner(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.tile_positions = None
        self.draw_grid_bool = True

    def build(
        self, preview: dai.Node.Output, nn: dai.Node.Output, tile_positions
    ) -> "QRScanner":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.tile_positions = tile_positions
        return self

    def process(self, preview, detections) -> None:
        frame = preview.getCvFrame()
        if self.draw_grid_bool:
            self.draw_grid(frame)

        box_color = (255, 87, 51)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        for det in detections.detections:
            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=box_color,
                thickness=2,
            )

            confidence_text = f"{int(det.confidence * 100)}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                confidence_text, font, font_scale, text_thickness
            )

            # background of text
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1] - text_height - baseline),
                (bbox[0] + text_width + 10, bbox[1]),
                box_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                frame,
                confidence_text,
                (bbox[0] + 5, bbox[1] - 5),
                font,
                font_scale,
                text_color,
                text_thickness,
            )

            if DECODE:
                decoded_text = self.decode(frame, bbox)
                if decoded_text:
                    (decoded_text_width, decoded_text_height), decoded_baseline = (
                        cv2.getTextSize(decoded_text, font, font_scale, text_thickness)
                    )

                    # background of text
                    cv2.rectangle(
                        frame,
                        (
                            bbox[0],
                            bbox[1]
                            - decoded_text_height
                            - baseline
                            - decoded_text_height
                            - decoded_baseline,
                        ),
                        (
                            bbox[0] + decoded_text_width + 10,
                            bbox[1] - text_height - baseline,
                        ),
                        box_color,
                        thickness=cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        decoded_text,
                        (bbox[0] + 5, bbox[1] - text_height - baseline - 5),
                        font,
                        font_scale,
                        text_color,
                        text_thickness,
                    )

        cv2.imshow("Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
        elif key == ord("g"):
            self.draw_grid_bool = not self.draw_grid_bool

    def decode(self, frame, bbox):
        """
        Decode the QR code present in the given bounding box.
        """
        assert DECODE
        if bbox[1] == bbox[3] or bbox[0] == bbox[2]:
            print("Detection bbox is empty")
            return ""

        bbox = expandBbox(bbox, frame, 5)  # expand bbox by 5%
        img = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        data = decode(img)
        if data:
            text = data[0].data.decode("utf-8")
            print("Decoded text", text)
            return text
        else:
            print("Decoding failed")
            return ""

    def draw_grid(self, frame: np.ndarray) -> None:
        if not self.tile_positions:
            print("Error: Tile positions are not set.")
            return

        img_height, img_width, _ = frame.shape

        np.random.seed(432)
        colors = [
            (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                0.3,
            )
            for _ in range(len(self.tile_positions))
        ]

        for idx, tile_info in enumerate(self.tile_positions):
            x1, y1, x2, y2 = tile_info["coords"]
            color = colors[idx % len(colors)]
            self._draw_filled_rect_with_alpha(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), color
            )

        grid_info_text = f"Tiles: {len(self.tile_positions)}"
        text_x = img_width // 2 - 100
        text_y = img_height - 30

        cv2.putText(
            frame,
            grid_info_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_filled_rect_with_alpha(
        self, frame, top_left, bottom_right, color_with_alpha
    ):
        overlay = frame.copy()
        output = frame.copy()
        color = color_with_alpha[:3]
        alpha = color_with_alpha[3]

        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        np.copyto(frame, output)


def expandBbox(bbox: np.ndarray, frame: np.ndarray, percentage: float) -> np.ndarray:
    """
    Expand the bounding box by a certain percentage.
    """
    bbox_copy = bbox.copy()
    pixels_expansion_0 = (bbox_copy[3] - bbox_copy[1]) * (percentage / 100)
    pixels_expansion_1 = (bbox_copy[2] - bbox_copy[0]) * (percentage / 100)
    bbox_copy[0] = max(0, bbox_copy[0] - pixels_expansion_1)
    bbox_copy[1] = max(0, bbox_copy[1] - pixels_expansion_0)
    bbox_copy[2] = min(frame.shape[1], bbox_copy[2] + pixels_expansion_1)
    bbox_copy[3] = min(frame.shape[0], bbox_copy[3] + pixels_expansion_0)
    return bbox_copy


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
