import cv2
import numpy as np
import tensorflow as tf
import posenet

from lp_utils.pose import TorsoKeypoints


def adapt_pic(print_, image, torso: TorsoKeypoints):
    """
    :param print_: picture to be transformed
    :param image: image to be augmented with pic
    :param torso: tuple(lsh, rsh, lh, rh)
        lsh: left shoulder coordinates
        rsh: right shoulder coordinates
        lh: left hip coordinates
        rh: right hip coordinates
    :return: transformed pic
    """
    x, y, *_ = print_.shape
    transformation_matrix = cv2.getPerspectiveTransform(
        np.float32(((0, 0), (x, 0), (0, y), (x, y))),
        np.float32(
            [
                torso.left_shoulder.coords(),
                torso.right_shoulder.coords(),
                torso.left_hip.coords(),
                torso.right_hip.coords(),
            ]
        ),
    )

    ix, iy, *_ = image.shape
    dim = (iy, ix)
    return cv2.warpPerspective(print_, transformation_matrix, dim)


def read_apng(filenames):
    return [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in filenames]


def overlay_transparent(background_img, img_to_overlay_t):
    """
    @brief      Overlays a transparent PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)

    @return     Background image with overlay on top
    """

    background_img = background_img.copy()

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, mask = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    # mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    # roi = background_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(
        background_img, background_img, mask=cv2.bitwise_not(mask)
    )

    # cv2.namedWindow('callibration', cv2.WND_PROP_FULLSCREEN)

    # Mask out the logo from the logo image.
    # img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    return cv2.add(img1_bg, overlay_color)


class ProjectableRegion:
    def __init__(self, ul, ur, dl, dr, output_width, output_height):
        self._coords = (ul, ur, dl, dr)
        self._output_resolution = (output_width, output_height)
        pts1 = np.float32(self._coords)
        pts2 = np.float32(
            (
                (0, 0),
                (output_width, 0),
                (0, output_height),
                (output_width, output_height),
            )
        )
        self._transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def of(self, webcam_img):
        return cv2.warpPerspective(
            webcam_img, self._transformation_matrix, self._output_resolution
        )


class Apng:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def next_frame(self):
        i = self.i
        self.i = (self.i + 1) % len(self.frames)
        return self.frames[i]
