import cv2
import numpy as np


class EffectRenderer2D:
    def __init__(self,
                 effect_image_path,
                 src_points_path="res/source_landmarks.npy",
                 filter_points_path="res/filter_points.npy"):
        self.effect_image = cv2.imread(effect_image_path, cv2.IMREAD_UNCHANGED)
        self.src_points = np.load(src_points_path)

        self.filter_points = np.load(filter_points_path)
        self.src_points = self.src_points[self.filter_points]

        self.subdiv = cv2.Subdiv2D((0, 0, self.effect_image.shape[1], self.effect_image.shape[0]))
        self.subdiv.insert(self.src_points.tolist())

        self.triangles = np.array([(self.src_points == value).all(axis=1).nonzero()
                                   for element in self.subdiv.getTriangleList()
                                   for value in element.reshape((3, 2))])
        self.triangles = self.triangles.reshape(len(self.triangles) // 3, 3)

    def render_effect(self, target_image, target_landmarks):
        ldms = target_landmarks[self.filter_points]
        effect = self.create_effect(target_image, ldms)
        return self.overlay_image(target_image, effect)

    def create_effect(self, target_image, dst_points):
        """
        Creates effect image that can be rendered on the target image
        :param target_image: target image
        :param dst_points: landmarks on the target image, should be of the same size and order as self.filter_points
        :return: effect image
        """

        # create empty overlay
        overlay = np.zeros((target_image.shape[0], target_image.shape[1], 4), np.uint8)

        for idx_tri in self.triangles:
            src_tri = self.src_points[idx_tri]
            dst_tri_full = dst_points[idx_tri]
            dst_tri = dst_tri_full[:, :2].astype(np.int32)

            src_tri_crop, src_crop = self.crop_triangle_bb(self.effect_image, src_tri)
            dst_tri_crop, overlay_crop = self.crop_triangle_bb(overlay, dst_tri)

            warp_mat = cv2.getAffineTransform(np.float32(src_tri_crop), np.float32(dst_tri_crop))
            warp = cv2.warpAffine(src_crop, warp_mat, (overlay_crop.shape[1], overlay_crop.shape[0]),
                                  None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask = np.zeros((overlay_crop.shape[0], overlay_crop.shape[1], 4), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_crop), (1.0, 1.0, 1.0, 1.0), 16, 0)
            mask[np.where(overlay_crop > 0)] = 0

            cropped_triangle = warp * mask
            overlay_crop += cropped_triangle

        return overlay

    def overlay_image(self, background_image, foreground_image, blur=0):
        """
        Take the two assets, and produce an image where foreground image overlays the background image
        :param background_image: background BRG or BGRA image with 0-255 values, transparency will be ignored in the result
        :param foreground_image: foreground BGRA image with 0-255 values
        :return: BGR image with foreground image overlaying the background image
        """

        mask = foreground_image[:, :, 3]

        if blur > 0:
            mask = cv2.medianBlur(mask, blur)

        mask_inv = 255 - mask

        overlayed = foreground_image[:, :, :3] * np.dstack([mask / 255.0] * 3) + \
                    background_image[:, :, :3] * np.dstack([mask_inv / 255.0] * 3)
        overlayed = overlayed.astype(np.uint8)

        return overlayed

    def crop_triangle_bb(self, image, triangle):
        """
        Create a trinagle bounding box and return cropped image.
        :param image: Target image
        :param triangle: Triangle coordinates (3x2 array)
        :return: Tupple (Triangle crop coordinates relative to the cropped image, cropped image)
        """
        x, y, w, h = cv2.boundingRect(triangle)
        crop = image[y:y + h, x:x + w]
        triangle_crop = triangle - np.array([x, y])
        return triangle_crop, crop
